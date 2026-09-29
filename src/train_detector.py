"""Train only on explicitly supplied labels. No labels inferred from ratings."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from .detector import evidence_features, normalized_url, assess
from .prepare_data import write_json


def vector(candidate, as_of):
    flags, *_ = evidence_features(candidate, as_of)
    return {key: float(value) for key,value in flags.items()}


def load_training(candidates, labels_path):
    by_id={c['candidate_id']:c for c in candidates}
    if len(by_id)!=len(candidates):
        raise ValueError('Duplicate candidate_id')
    rows, labels, origins, reviewers, seen = [], [], set(), set(), set()
    with labels_path.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if not row.get('label','').strip():
                continue
            cid=row['candidate_id']
            if cid in seen or cid not in by_id:
                raise ValueError(f'Duplicate or unknown labeled candidate: {cid}')
            seen.add(cid)
            if row['label'] not in ('0','1'):
                raise ValueError('Labels must be 0 or 1; leave undecided labels empty')
            if row.get('label_origin') not in ('internal_human','organizer','synthetic_test'):
                raise ValueError('Explicit label_origin is required')
            if not row.get('reviewer','').strip() or not row.get('rationale','').strip():
                raise ValueError('Each label needs a reviewer and rationale')
            rows.append(by_id[cid]);labels.append(int(row['label']))
            origins.add(row['label_origin']);reviewers.add(row['reviewer'])
    counts=Counter(labels)
    if len(counts)!=2 or min(counts.values())<4:
        raise ValueError('Need at least 4 explicitly labeled examples of each class. Current workbook ratings are not binary labels.')
    if 'synthetic_test' in origins and len(origins)>1:
        raise ValueError('Do not mix synthetic examples with real evaluation labels')
    return rows, np.array(labels), sorted(origins), sorted(reviewers)


def merged_groups(rows):
    parent=list(range(len(rows)))
    def find(i):
        if parent[i]!=i: parent[i]=find(parent[i])
        return parent[i]
    seen={}
    for i,c in enumerate(rows):
        group=c.get('technology_group_id')
        if not isinstance(group,str) or not group.strip():
            raise ValueError('Reviewed technology_group_id required for every labeled candidate')
        keys=[('topic',group)] + [('url',normalized_url(s['url'])) for s in c['sources']]
        for key in keys:
            if key in seen: parent[find(i)]=find(seen[key])
            seen[key]=i
    return np.array([find(i) for i in range(len(rows))])


def metrics(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)), 'precision_positive':float(p[1]),
            'recall_positive':float(r[1]), 'f1_positive':float(f[1]), 'macro_f1':float(f.mean()),
            'confusion_matrix_labels_0_1':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def train(candidates, labels_path, as_of, output):
    rows,y,origins,reviewers=load_training(candidates,labels_path)
    groups=merged_groups(rows)
    if min(len(set(groups[y==k])) for k in (0,1))<3:
        raise ValueError('Need at least 3 independent groups supporting each class for 3-fold evaluation')
    X=[vector(c,as_of) for c in rows]
    if any(not any(x.values()) for x in X):
        raise ValueError('Labeled rows need eligible reviewed evidence; empty feature vectors found')
    predictions=np.zeros(len(y),dtype=int);baselines=np.zeros(len(y),dtype=int);folds=np.zeros(len(y),dtype=int)
    for fold,(tr,te) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(X,y,groups)):
        if len(set(y[tr]))<2:
            raise ValueError('A training fold contains one class. Add independent labeled groups.')
        assert not set(groups[tr]) & set(groups[te])
        model=make_pipeline(DictVectorizer(),LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=42))
        model.fit([X[i] for i in tr],y[tr])
        predictions[te]=model.predict([X[i] for i in te]);folds[te]=fold
        baselines[te]=int(np.mean(y[tr])>0.5)
    final=make_pipeline(DictVectorizer(),LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=42))
    final.fit(X,y)
    vectorizer,estimator=final.steps[0][1],final.steps[1][1]
    output.mkdir(parents=True,exist_ok=True)
    artifact={'model_version':'binary-evidence-logistic-0.1','as_of':as_of,'label_origins':origins,
              'synthetic_test_only':origins==['synthetic_test'],'calibration':'not_calibrated',
              'weights':dict(zip(vectorizer.get_feature_names_out(),map(float,estimator.coef_[0]))),
              'intercept':float(estimator.intercept_[0]), 'threshold':0.5}
    write_json(output/'classifier.json',artifact)
    report={'task':'binary_classification_of_reviewed_evidence','label_origins':origins,'reviewers':reviewers,
            'synthetic_test_only':origins==['synthetic_test'], 'as_of':as_of,
            'count':len(y),'groups':len(set(groups)), 'class_counts':{str(k):int(sum(y==k)) for k in (0,1)},
            'metrics':metrics(y,predictions),'majority_baseline':metrics(y,baselines),
            'protocol':'3-fold StratifiedGroupKFold, seed 42, C=1, balanced weights, no tuning, threshold 0.5',
            'labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            'candidates_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'limitations':['Metrics apply to supplied reviewed labels only, not automatically organizer evaluation.',
                           'Model learns curated evidence features; extraction quality is not evaluated.',
                           'No independent external test or probability calibration.',
                           'Semantic/topic annotation may leak expectations; require independent reviewers.']}
    write_json(output/'binary_evaluation.json',report)
    with (output/'binary_oof.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['candidate_id','group','fold','label','prediction','majority'])
        writer.writerows((c['candidate_id'],int(groups[i]),int(folds[i]),int(y[i]),int(predictions[i]),int(baselines[i])) for i,c in enumerate(rows))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--as-of',required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/binary'))
    args=parser.parse_args()
    data=json.loads(args.input.read_text(encoding='utf-8'))
    result=train(data['candidates'] if isinstance(data,dict) else data,args.labels,args.as_of,args.output)
    print(json.dumps(result['metrics']))


if __name__=='__main__':main()
