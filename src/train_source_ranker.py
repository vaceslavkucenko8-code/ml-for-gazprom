"""Group-validation on original ratings using only retrieved source features."""
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .source_ranker import features,predict
from .train_baseline import group_ids


def run(root=Path('.')):
    rows=json.loads((root/'data/processed/signals_clean.json').read_text(encoding='utf-8'))
    linked=json.loads((root/'data/enrichment/linked_sources.json').read_text(encoding='utf-8'))
    by_id={r['signal_id']:r for r in linked}
    used=[];vectors=[];documents=[];snapshots={}
    as_of='2026-09-20'
    for row in rows:
        docs=[]
        for link in by_id[row['signal_id']]['sources']:
            if link['status']!='retrieved_unverified':continue
            path=root/'data/enrichment'/link['cache_file']
            record=json.loads(path.read_text(encoding='utf-8'))
            docs.append(record['document'])
            snapshots[link['cache_file']]=hashlib.sha256(path.read_bytes()).hexdigest()
        if docs:
            used.append(row);documents.append(docs);vectors.append(features(docs,as_of))
    names=sorted(vectors[0]);x=np.array([[v[k] for k in names] for v in vectors])
    y=np.array([r['reference_score'] for r in used],dtype=float)
    config=json.loads((root/'data/technology_groups.json').read_text(encoding='utf-8'))
    all_groups=dict(zip([r['signal_id'] for r in rows],group_ids(rows,config)))
    groups=np.array([all_groups[r['signal_id']] for r in used])
    predictions=np.zeros(len(y));baseline=np.zeros(len(y));fold_ids=np.zeros(len(y),dtype=int)
    for fold,(train,test) in enumerate(GroupKFold(5).split(x,y,groups)):
        model=make_pipeline(StandardScaler(),Ridge(alpha=10))
        model.fit(x[train],y[train]);predictions[test]=np.clip(model.predict(x[test]),3,7)
        baseline[test]=np.median(y[train]);fold_ids[test]=fold
    mae=float(np.abs(y-predictions).mean());base_mae=float(np.abs(y-baseline).mean())
    final=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,y)
    scaler=final[0];regressor=final[1]
    weights=regressor.coef_/scaler.scale_
    artifact={'model_version':'source-priority-ridge-0.1','as_of':as_of,
              'intercept':float(regressor.intercept_-np.dot(weights,scaler.mean_)),
              'weights':dict(zip(names,map(float,weights))),
              'deployment_enabled':mae<base_mae,
              'target':'reference_score; not a weak-signal class','training_rows':len(used),
              'snapshot_hashes':snapshots}
    assert np.allclose([predict(d,artifact,as_of)['raw_score'] for d in documents],final.predict(x))
    report={'task':'source_based_reference_priority_regression','rows':len(used),'groups':len(set(groups)),
            'cv':'5-fold groups by technology families and shared URLs, alpha=10 fixed, no tuning',
            'mae':mae,'median_baseline_mae':base_mae,'deployment_enabled':artifact['deployment_enabled'],
            'binary_precision':None,'binary_recall':None,'binary_f1':None,
            'limitations':['Targets are ratings, not hidden organizer labels.',
                          'Source text snapshots may differ from text used to build workbook.',
                          'This compares rank-priority proxies, not binary detection accuracy.'],
            'oof':[{'signal_id':r['signal_id'],'group':int(groups[i]),'fold':int(fold_ids[i]),
                    'target':float(y[i]),'prediction':float(predictions[i]),'baseline':float(baseline[i])} for i,r in enumerate(used)]}
    for name,value in [('source_ranker.json',artifact),('source_ranker_evaluation.json',report)]:
        (root/'artifacts'/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('oof','limitations')}))


if __name__=='__main__':run()
