from src.source_ranker import features,predict


def test_no_reference_labels_in_source_features():
    doc={'text':'A prototype was tested.','published_at':'2026-09-01'}
    a=features([doc],'2026-09-20')
    assert a==features([doc|{'reference_score':7,'rationale_raw':'weak','signal_id':1}],'2026-09-20')
    assert a['early']==1


def test_contributions_reconstruct_priority_without_probability():
    model={'weights':{'early':0.5},'intercept':4,'deployment_enabled':True,'model_version':'test'}
    result=predict([{'text':'Prototype trial.'}],model,'2026-09-20')
    assert result['score']==4.5
    assert result['probability_weak'] is None
    assert sum(c['contribution'] for c in result['contributions'])==0.5
