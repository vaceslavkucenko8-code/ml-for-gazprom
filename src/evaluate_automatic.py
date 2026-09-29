"""Evaluate explicit three-way annotations, keeping abstention visible."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .automatic import analyse, VERSION

LABELS=('likely_weak','likely_mature','needs_review')


def evaluate(corpus):
    cases=corpus['cases']
    if not cases or len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Require nonempty cases with unique IDs')
    matrix={expected:{actual:0 for actual in LABELS} for expected in LABELS}
    rows=[]
    for case in cases:
        if case.get('expected') not in LABELS or not case.get('rationale') or not case.get('annotation_origin'):
            raise ValueError('Explicit annotation, origin and rationale required')
        result=analyse(case['candidate'],case['as_of'],case.get('search_query'))
        predicted=result['decision'];expected=case['expected']
        matrix[expected][predicted]+=1
        rows.append({'case_id':case['case_id'],'expected':expected,'predicted':predicted,
                     'matches_annotation':predicted==expected,'reason':result['reason'],
                     'missing_features':result['missing_features'],
                     'sources':[s['url'] for s in case['candidate']['sources']],
                     'claims':result['claims']})
    per_class={}
    for label in LABELS:
        tp=matrix[label][label];actual=sum(matrix[label].values());predicted=sum(matrix[r][label] for r in LABELS)
        precision=tp/predicted if predicted else None
        recall=tp/actual if actual else None
        f1=2*tp/(actual+predicted) if actual+predicted else None
        per_class[label]={'support':actual,'predicted':predicted,'precision':precision,'recall':recall,'f1':f1}
    decided=[r for r in rows if r['predicted']!='needs_review']
    return {'model_version':VERSION,'annotation_kind':corpus.get('annotation_kind'),
            'independent_test':bool(corpus.get('independent_test',False)),
            'case_count':len(rows),'annotation_distribution':dict(Counter(c['expected'] for c in cases)),
            'abstention_rate':1-len(decided)/len(rows),'decision_coverage':len(decided)/len(rows),
            'agreement_with_annotations':sum(r['matches_annotation'] for r in rows)/len(rows),
            'agreement_on_decided':sum(r['matches_annotation'] for r in decided)/len(decided) if decided else None,
            'per_class':per_class,'confusion_matrix':matrix,'cases':rows,
            'limitations':['Provisional AI-reviewed development cases; not organizer labels or expert validation.',
                          'Cases used to change rules are not independent evidence of generalization.',
                          'Needs_review is abstention, never a negative binary training label.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('data/validation/corpus.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/automatic_evaluation.json'))
    args=parser.parse_args()
    raw=args.input.read_bytes();report=evaluate(json.loads(raw))
    report['corpus_sha256']=hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('case_count','abstention_rate','agreement_with_annotations','annotation_distribution')}))


if __name__=='__main__':
    main()
