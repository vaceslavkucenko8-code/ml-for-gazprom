"""Convert participant 2 exports without promoting heuristics to reviewed facts."""
import argparse
import json
from pathlib import Path
from .detector import FEATURES

TYPE_MAP = {'primary_official': 'official', 'academic_paper': 'research', 'university': 'university',
            'patent': 'patent', 'industry_media': 'professional_media',
            'press_release': 'press_release', 'aggregator': 'aggregator',
            'personal_blog': 'blog'}


def adapt_export(payload, reviews=None):
    documents = payload['documents']
    docs = {d['doc_id']: d for d in documents}
    if len(docs) != len(documents):
        raise ValueError('Duplicate doc_id')
    candidates = payload['candidates']
    if len({c['candidate_id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate_id')
    reviews = reviews or []
    used_reviews = set()
    result = []
    for candidate in candidates:
        cid = candidate['candidate_id']
        sources, warnings = [], []
        for sid in dict.fromkeys(candidate['source_doc_ids']):
            if sid not in docs:
                warnings.append(f'Missing document: {sid}')
                continue
            d = docs[sid]
            if not d.get('text', '').strip() or not d.get('title', '').strip():
                warnings.append(f'Empty document: {sid}')
                continue
            sources.append({'source_id': sid, 'url': d['url'],
                'title_original': d.get('original_title') or d['title'], 'text': d['text'],
                'published_at': None if d.get('published_at_is_estimated') else (d.get('published_at') or '')[:10] or None,
                'source_type': TYPE_MAP.get(d.get('source_type'), 'unknown'),
                'trust_level': d.get('trust_level') or 'unknown',
                'trust_reason': d.get('trust_explanation') or '',
                'language': d.get('language'), 'independence_reviewed': False,
                'independence_group': None, 'retrieval_metadata': d})
        source_ids = {s['source_id'] for s in sources}
        evidence_by_id = {e['evidence_id']: e for e in candidate.get('evidence', [])}
        if len(evidence_by_id) != len(candidate.get('evidence', [])):
            raise ValueError('Duplicate evidence_id')
        assertions = []
        assertion_keys = set()
        for index, review in enumerate(reviews):
            if review['candidate_id'] != cid:
                continue
            e = evidence_by_id.get(review['evidence_id'])
            if not e or e['doc_id'] not in source_ids:
                raise ValueError('Review references missing evidence/source')
            if review.get('feature') not in FEATURES:
                raise ValueError('Unknown feature')
            key = (review['evidence_id'], review['feature'])
            if key in assertion_keys:
                raise ValueError('Duplicate evidence/feature review')
            assertion_keys.add(key)
            if not review.get('reviewer', '').strip() or not review.get('rationale', '').strip():
                raise ValueError('Review requires reviewer and rationale')
            d = docs[e['doc_id']]
            if e.get('is_generated_summary') or d.get('is_generated_summary'):
                raise ValueError('Generated summary cannot be original evidence')
            if not e.get('quote') or e['quote'] not in d['text']:
                raise ValueError('Quote does not match document')
            if e.get('is_future_looking') and review.get('event_status') == 'observed':
                raise ValueError('Future claim cannot be promoted to observed')
            assertions.append({**review, 'evidence_id': f"{e['evidence_id']}:{review['feature']}",
                'source_id': e['doc_id'], 'quote': e['quote'],
                'reviewed': review.get('reviewed') is True,
                'scope_matches_candidate': review.get('scope_matches_candidate') is True})
            used_reviews.add(index)
        result.append({'candidate_id': cid, 'name_ru': candidate['technology_name'],
            'query': candidate['query'], 'technology_group_id': cid,
            'sources': sources, 'evidence': assertions,
            'review_status': 'review_imported' if assertions else 'pending_semantic_review',
            'review_warnings': warnings,
            'name_requires_russian_editorial_review': True,
            'extracted_evidence_pending': list(evidence_by_id.values()),
            'participant2_candidate': candidate})
    if len(used_reviews) != len(reviews):
        raise ValueError('Review references unknown candidate')
    return {'synthetic': payload.get('synthetic', False),
            'provenance': 'participant2_export; source authenticity not verified by adapter',
            'candidates': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviews', type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    reviews = json.loads(args.reviews.read_text(encoding='utf-8')) if args.reviews else []
    output = adapt_export(payload, reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Adapted {len(output['candidates'])} candidates")


if __name__ == '__main__':
    main()
