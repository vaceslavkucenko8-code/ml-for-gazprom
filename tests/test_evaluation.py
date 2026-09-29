import pytest
from src.evaluate_automatic import evaluate
from test_automatic import candidate


def case(text,expected,key='case'):
    return {'case_id':key,'candidate':candidate(text),'as_of':'2026-09-20','expected':expected,
            'annotation_origin':'synthetic_test','rationale':'Software fixture, not a real label'}


def test_abstention_is_counted_in_coverage_not_hidden():
    data={'cases':[case('This quantum sensor prototype was demonstrated.','likely_weak'),
                   case('This quantum sensor is widely deployed in industry.','likely_mature','second')]}
    r=evaluate(data)
    assert r['decision_coverage']==0.5
    assert r['abstention_rate']==0.5
    assert r['per_class']['likely_weak']['recall']==0
    assert r['per_class']['likely_weak']['precision'] is None
    assert r['agreement_on_decided']==1
    assert r['agreement_with_annotations']==0.5


def test_empty_or_duplicate_corpus_rejected():
    with pytest.raises(ValueError):evaluate({'cases':[]})
    c=case('This quantum sensor is a prototype.','needs_review')
    with pytest.raises(ValueError):evaluate({'cases':[c,c]})


def test_explicit_standard_without_mass_adoption():
    c=candidate('These finalized quantum sensor standards are ready for implementation.')
    r=evaluate({'cases':[{'case_id':'standard','candidate':c,'as_of':'2026-09-20',
        'expected':'likely_mature','annotation_origin':'synthetic_test','rationale':'Completed standard'}]})
    assert r['cases'][0]['predicted']=='likely_mature'
    assert all(e['feature']!='mass_adoption' for e in r['cases'][0]['claims'])


def test_institutional_news_is_not_mislabeled_as_paper():
    from src.retrieval.parse import guess_source_type
    from src.retrieval.schemas import SourceType
    assert guess_source_type('https://news.mit.edu/2025/example')[0]==SourceType.UNIVERSITY
    assert guess_source_type('https://news.mit.edu.evil.example/')[0]!=SourceType.UNIVERSITY
