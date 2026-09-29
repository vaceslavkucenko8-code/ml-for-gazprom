import pytest
from src.self_review_evaluation import metrics


def test_abstention_and_unresolved_labels_are_not_successful_predictions():
    labels=[{'signal_id':1,'label':'weak','binary_label':1},
            {'signal_id':2,'label':'reject','binary_label':0},
            {'signal_id':3,'label':'needs_review','binary_label':None}]
    r=metrics({'1':'needs_review','2':'likely_mature','3':'likely_weak'},labels)
    assert r['recall_resolved']==0
    assert r['precision_resolved'] is None
    assert r['decision_coverage_resolved']==.5
    assert r['correct_decisions_fraction_resolved']==.5
    assert r['unresolved_called_weak']==1


def test_missing_predictions_are_rejected():
    with pytest.raises(ValueError):metrics({},[{'signal_id':1,'label':'weak','binary_label':1}])


def test_uncertainty_cannot_be_converted_to_negative_label():
    with pytest.raises(ValueError):metrics({'1':'likely_mature'},[{'signal_id':1,'label':'needs_review','binary_label':0}])
