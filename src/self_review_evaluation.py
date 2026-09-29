"""Reproducible development evaluation against explicitly AI-authored labels.

Unresolved labels are never converted to negative examples. Abstentions on
resolved positives count as missed positives. No claim of a blind test is made.
"""
import argparse
import json
import importlib.util
import hashlib
from collections import Counter
from pathlib import Path
from .automatic import analyse, VERSION
from .blind_evaluation import digest
from .prepare_self_review import prepare


def metrics(predictions, labels):
    if len({r['signal_id'] for r in labels})!=len(labels):
        raise ValueError('Duplicate label IDs')
    if set(predictions)!={str(r['signal_id']) for r in labels}:
        raise ValueError('Predictions must cover every annotation')
    matrix={label:{decision:0 for decision in ('likely_weak','likely_mature','needs_review')}
            for label in ('weak','reject','needs_review')}
    for r in labels:
        if r['label'] not in matrix: raise ValueError('Unknown label')
        if r['binary_label']!={'weak':1,'reject':0,'needs_review':None}[r['label']]:
            raise ValueError('Unresolved labels must remain null')
        matrix[r['label']][predictions[str(r['signal_id'])]]+=1
    tp=matrix['weak']['likely_weak']; fp=matrix['reject']['likely_weak']
    positive=sum(matrix['weak'].values()); negative=sum(matrix['reject'].values()); n=positive+negative
    tn=matrix['reject']['likely_mature']; abstain=matrix['weak']['needs_review']+matrix['reject']['needs_review']
    fn=positive-tp
    return {'n_all':len(labels),'n_binary_resolved':n,'annotation_coverage':n/len(labels) if labels else None,
            'confusion':matrix,'precision_resolved':tp/(tp+fp) if tp+fp else None,
            'recall_resolved':tp/positive if positive else None,
            'f1_resolved':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'decision_coverage_resolved':(n-abstain)/n if n else None,
            'correct_decisions_fraction_resolved':(tp+tn)/n if n else None,
            'unresolved_called_weak':matrix['needs_review']['likely_weak'],
            'all_positive_baseline_correct_fraction_resolved':positive/n if n else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.'));args=parser.parse_args();root=args.root
    corpus=json.loads((root/'data/self_review/corpus.json').read_text(encoding='utf-8'))
    annotation=json.loads((root/'data/self_review/labels.json').read_text(encoding='utf-8'))
    frozen=json.loads((root/'artifacts/self-review/baseline/frozen.json').read_text(encoding='utf-8'))
    if digest(frozen['payload'])!=frozen['sha256']:raise ValueError('Baseline snapshot changed')
    if frozen['payload']['cases']!=corpus['cases']:raise ValueError('Evaluation inputs changed')
    before={p['case_id']:p['decision'] for p in frozen['payload']['predictions']}
    after={};details=[]
    for case in corpus['cases']:
        result=analyse(case['candidate'],case['as_of']); after[case['case_id']]=result['decision']
        details.append(result)
    labels=annotation['labels']
    prepared=prepare(root)
    if hashlib.sha256((root/'artifacts/self-review/baseline/automatic.py').read_bytes()).hexdigest()!=frozen['payload']['snapshot']['src/automatic.py']:
        raise ValueError('Baseline code changed')
    spec=importlib.util.spec_from_file_location('frozen_automatic',root/'artifacts/self-review/baseline/automatic.py')
    baseline_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline_module)
    prepared_before={};prepared_after={}
    for case in prepared['cases']:
        prepared_before[case['case_id']]=baseline_module.analyse(case['candidate'],case['as_of'])['decision']
        prepared_after[case['case_id']]=analyse(case['candidate'],case['as_of'])['decision']
    result={'origin':'internal_ai','status':'development_only','independent_holdout':False,
            'model_version':VERSION,'input_sha256':digest(corpus),'labels_sha256':digest(annotation),
            'baseline_sha256':frozen['sha256'],'baseline':metrics(before,labels),'current':metrics(after,labels),
            'prediction_counts':dict(Counter(after.values())),
            'prepared_input_sha256':digest(prepared),
            'prepared_baseline':metrics(prepared_before,labels),
            'prepared_current':metrics(prepared_after,labels),
            'limitations':['Labels are AI judgments with unresolved cases and provisional positive scope.',
                'The same assistant labeled and fixed code: no independent generalization claim.',
                'Evaluation uses cached source metadata unchanged, including unknown dates/trust.',
                'Russian query overlap and missing source metadata reduce rule coverage.',
                'This measures classification of supplied candidates, not open-search recall.'],
            'rows':[{'signal_id':r['signal_id'],'label':r['label'],'before':before[str(r['signal_id'])],
                     'after':after[str(r['signal_id'])],
                     'prepared_before':prepared_before[str(r['signal_id'])],
                     'prepared_after':prepared_after[str(r['signal_id'])]} for r in labels]}
    out=root/'artifacts/self-review';out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('prepared_baseline','prepared_current','prediction_counts')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
