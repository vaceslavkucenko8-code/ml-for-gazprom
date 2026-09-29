"""Freeze predictions before labels arrive; score a complete external label set."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from .automatic import analyse, VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze(corpus):
    cases = corpus['cases']
    ids = [c['case_id'] for c in cases]
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique case IDs required')
    clean = [{k: c[k] for k in ('case_id', 'candidate', 'as_of', 'search_query') if k in c} for c in cases]
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'src').rglob('*.py')) + [root/'artifacts/source_ranker.json', root/'requirements.txt']
    snapshot = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.exists()}
    payload = {'created_at': datetime.now(timezone.utc).isoformat(), 'model_version': VERSION,
               'scope': 'classification_of_supplied_candidates_not_end_to_end_search_recall',
               'dataset_role': corpus.get('dataset_role', 'unspecified'),
               'labels_present_at_freeze': any('expected' in c or 'label' in c for c in cases),
               'snapshot': snapshot, 'cases': clean, 'predictions': []}
    for c in clean:
        r = analyse(c['candidate'], c['as_of'], c.get('search_query'))
        payload['predictions'].append({'case_id': c['case_id'], 'decision': r['decision'], 'reason': r['reason']})
    return {'sha256': digest(payload), 'payload': payload}


def score(bundle, labels):
    p = bundle['payload']
    if digest(p) != bundle['sha256']:
        raise ValueError('Frozen package changed')
    if labels.get('prediction_sha256') != bundle['sha256']:
        raise ValueError('Labels must reference this frozen package')
    rows = labels['labels']
    expected_ids = [r['case_id'] for r in p['predictions']]
    supplied = [r['case_id'] for r in rows]
    if len(set(supplied)) != len(supplied) or set(supplied) != set(expected_ids):
        raise ValueError('Require exactly one label for EVERY frozen case')
    if any(type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Explicit integer binary labels required')
    if not labels.get('provenance', '').strip():
        raise ValueError('Label provenance required')
    truth = {r['case_id']: r['label'] for r in rows}
    matrix = {'0': {'positive': 0, 'negative': 0, 'abstain': 0}, '1': {'positive': 0, 'negative': 0, 'abstain': 0}}
    mapping = {'likely_weak': 'positive', 'likely_mature': 'negative', 'needs_review': 'abstain'}
    for r in p['predictions']:
        matrix[str(truth[r['case_id']])][mapping[r['decision']]] += 1
    tp, fp = matrix['1']['positive'], matrix['0']['positive']
    fn = matrix['1']['negative'] + matrix['1']['abstain']
    abstain = matrix['0']['abstain'] + matrix['1']['abstain']
    n = len(rows)
    eligible = (p['dataset_role'] == 'held_out' and not p['labels_present_at_freeze']
                and labels.get('origin') == 'organizer' and labels.get('held_out_attested') is True)
    return {'prediction_sha256': bundle['sha256'], 'labels_sha256': digest(labels), 'n': n,
            'scope': p['scope'], 'confusion': matrix,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'decision_coverage': (n-abstain)/n, 'abstention_rate': abstain/n,
            'correct_decisions_fraction': (tp+matrix['0']['negative'])/n,
            'label_origin': labels.get('origin'), 'provenance': labels['provenance'],
            'status': 'external_holdout_reported_by_provider' if eligible else 'development_only',
            'limitations': ['Holdout independence depends on provider provenance, not a software flag.',
                            'Positive abstentions count as missed positives; no cases are dropped.',
                            'Candidate classification does not measure missed candidates in open search.',
                            'Hashes detect changes relative to this package, not provide a trusted timestamp.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'score'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    result = freeze(data) if args.action == 'freeze' else score(data, json.loads(args.labels.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(args.out), 'sha256': result.get('sha256'), 'status': result.get('status', 'frozen_awaiting_labels')}))


if __name__ == '__main__':
    main()
