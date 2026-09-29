"""Exploratory text baseline on AI labels; never a production/organizer model."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from .train_detector import merged_groups, metrics
from .blind_evaluation import digest


def train(root=Path('.')):
    data=json.loads((root/'data/self_review/labels.json').read_text(encoding='utf-8'))
    corpus=json.loads((root/'data/self_review/corpus.json').read_text(encoding='utf-8'))
    by_id={int(c['case_id']):c['candidate'] for c in corpus['cases']}
    labels=[r for r in data['labels'] if r['binary_label'] is not None]
    # Shared sources and existing topic groups cannot cross validation folds.
    group_data=json.loads((root/'data/technology_groups.json').read_text(encoding='utf-8'))
    group_map={str(i):str(min(family['ids'])) for family in group_data['families'] for i in family['ids']}
    all_rows=list(by_id.values())
    for row in all_rows:
        row['technology_group_id']=group_map.get(row['candidate_id'],row['candidate_id'])
    all_groups=merged_groups(all_rows)
    id_to_group={row['candidate_id']:int(group) for row,group in zip(all_rows,all_groups)}
    rows=[by_id[r['signal_id']] for r in labels]
    groups=np.array([id_to_group[row['candidate_id']] for row in rows])
    texts=['\n'.join(s['title_original']+'\n'+s['text'][:6000] for s in row['sources']) for row in rows]
    y=np.array([r['binary_label'] for r in labels]); pred=np.zeros(len(y),int); base=np.zeros(len(y),int)
    folds=[]
    def model():
        return make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=5000,sublinear_tf=True),
            LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42))
    for index,(tr,te) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(texts,y,groups)):
        if len(set(y[tr]))!=2:raise ValueError('Insufficient class diversity in training fold')
        assert not set(groups[tr])&set(groups[te])
        estimator=model();estimator.fit([texts[i] for i in tr],y[tr])
        pred[te]=estimator.predict([texts[i] for i in te]);base[te]=int(y[tr].mean()>.5)
        folds.append({'fold':index,'train_ids':[rows[i]['candidate_id'] for i in tr],
                      'test_ids':[rows[i]['candidate_id'] for i in te]})
    final=model();final.fit(texts,y);vec,clf=final.steps[0][1],final.steps[1][1]
    out=root/'artifacts/self-review'
    artifact={'model_version':'ai-label-text-logistic-0.1','origin':'internal_ai','experimental_only':True,
        'production_enabled':False,'calibrated':False,'vocabulary':{k:int(v) for k,v in vec.vocabulary_.items()},
        'idf':vec.idf_.tolist(),'weights':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),
        'parameters':{'ngram_range':[1,2],'max_features':5000,'sublinear_tf':True,'C':1,'class_weight':'balanced'}}
    (out/'experimental_text_classifier.json').write_text(json.dumps(artifact,ensure_ascii=False),encoding='utf-8')
    report={'origin':'internal_ai','independent_holdout':False,'n':len(y),'groups':len(set(groups)),
        'label_counts':{str(k):int(sum(y==k)) for k in (0,1)},'oof_metrics':metrics(y,pred),
        'majority_baseline':metrics(y,base),'folds':folds,'labels_sha256':digest(data),
        'protocol':'3-fold StratifiedGroupKFold seed 42; TF-IDF fit only inside training fold; fixed C=1; no tuning; source text only; original rating and annotation rationale excluded',
        'limitations':['Only 16 provisionally resolved AI annotations; 84 unresolved rows excluded explicitly.',
            'Three negative examples make estimates unstable; this is not evidence of external accuracy.',
            'Same assistant annotated and implemented the experiment; not a blind independent test.',
            'Research writing style can act as a shortcut; never replace source-based evidence with this score.'],
        'rows':[{'signal_id':labels[i]['signal_id'],'label':int(y[i]),'oof_prediction':int(pred[i]),'correct':bool(y[i]==pred[i])} for i in range(len(y))]}
    (out/'text_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('n','groups','oof_metrics','majority_baseline')},indent=2))


if __name__=='__main__':train()
