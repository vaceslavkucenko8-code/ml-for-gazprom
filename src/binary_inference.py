"""Optional learned classifier; factual gates still determine eligibility."""
import math
from .detector import evidence_features, assess


def predict_with_classifier(candidate, model, as_of):
    if model.get('synthetic_test_only'):
        raise ValueError('Synthetic-test model cannot be used as a real detector')
    result=assess(candidate,as_of)
    flags,*_=evidence_features(candidate,as_of)
    z=model['intercept']+sum(float(v)*model['weights'].get(k,0) for k,v in flags.items())
    logistic=1/(1+math.exp(-max(-700,min(700,z))))
    result['classifier_score_uncalibrated']=logistic
    result['classifier_version']=model['model_version']
    result['classifier_label_origins']=model['label_origins']
    if result['decision']=='weak' and logistic<model.get('threshold',0.5):
        result['decision']='needs_review'
        result['explanation_ru']='Фактические критерии выполнены, но обученный классификатор не поддержал отбор. Требуется проверка.'
    # A classifier cannot override maturity rejection or lack of evidence.
    return result
