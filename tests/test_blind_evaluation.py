import copy
import pytest
from src.blind_evaluation import freeze, score, digest
from test_automatic import candidate


def fixture():
    bundle = freeze({'dataset_role': 'development', 'cases': [
        {'case_id': str(i), 'candidate': candidate('A quantum sensor prototype.'), 'as_of': '2026-09-20'} for i in range(4)]})
    for row, decision in zip(bundle['payload']['predictions'], ['likely_weak', 'likely_weak', 'needs_review', 'likely_mature']):
        row['decision'] = decision
    bundle['sha256'] = digest(bundle['payload'])
    labels = {'prediction_sha256': bundle['sha256'], 'origin': 'synthetic_test', 'provenance': 'Software fixture only',
              'labels': [{'case_id': str(i), 'label': y} for i, y in enumerate([1, 0, 1, 0])]}
    return bundle, labels


def test_abstentions_not_removed_from_recall():
    b, l = fixture()
    r = score(b, l)
    assert r['precision'] == .5 and r['recall'] == .5 and r['f1'] == .5
    assert r['decision_coverage'] == .75
    assert r['status'] == 'development_only'


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unknown', 'float', 'wrong_hash', 'tamper'])
def test_invalid_evaluation_rejected(change):
    b, l = fixture()
    if change == 'missing': l['labels'].pop()
    if change == 'duplicate': l['labels'].append(copy.deepcopy(l['labels'][0]))
    if change == 'unknown': l['labels'][0]['case_id'] = 'alien'
    if change == 'float': l['labels'][0]['label'] = 1.0
    if change == 'wrong_hash': l['prediction_sha256'] = 'wrong'
    if change == 'tamper': b['payload']['predictions'][0]['decision'] = 'needs_review'
    with pytest.raises(ValueError): score(b, l)


def test_visible_labels_prevent_holdout_claim():
    b, l = fixture()
    b['payload']['dataset_role'] = 'held_out'
    b['payload']['labels_present_at_freeze'] = True
    b['sha256'] = l['prediction_sha256'] = digest(b['payload'])
    l.update(origin='organizer', held_out_attested=True)
    assert score(b, l)['status'] == 'development_only'
