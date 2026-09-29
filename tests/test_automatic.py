import pytest
from src.automatic import analyse


def candidate(text,**source_changes):
    source={'source_id':'s','url':'https://example.org','title_original':'Quantum sensor prototype',
            'text':text,'published_at':'2026-09-01','source_type':'research','trust_level':'high','trust_reason':'Test fixture'}
    return {'candidate_id':'c','name_ru':'Квантовый сенсор','query':'quantum sensor','sources':[source|source_changes],'evidence':[]}


def test_explicit_early_limited_relevant_is_automatic_not_reviewed():
    c=candidate('This quantum sensor prototype operates only in a laboratory.')
    # "only in a laboratory" deliberately differs from supported pattern below.
    c['sources'][0]['text']='This quantum sensor prototype is limited to laboratory experiments.'
    result=analyse(c,'2026-09-20')
    assert result['decision']=='likely_weak'
    assert all(e['human_reviewed'] is False for e in result['claims'])
    assert c['evidence']==[]
    for e in result['claims']:
        assert c['sources'][0]['text'][e['start']:e['end']]==e['quote']


@pytest.mark.parametrize('text',[
    'This quantum sensor prototype will enter mass production.',
    'This quantum sensor is not widely deployed.',
    'Unlike existing quantum sensors in mass production, our sensor is a prototype.',
    'This quantum sensor is not an industry standard.',
    'Mass production of this quantum sensor has not begun.',
])
def test_plans_negations_and_comparisons_not_maturity(text):
    r=analyse(candidate(text),'2026-09-20')
    assert r['decision']!='likely_mature'
    assert not r['flags']['mass_adoption']
    assert not r['flags']['industry_standard']


def test_pilot_does_not_prove_limited_market():
    assert analyse(candidate('This quantum sensor prototype was tested in a pilot project.'),'2026-09-20')['decision']=='needs_review'


def test_observed_demonstration_is_not_erased_by_future_continuation():
    text='The researchers demonstrated a quantum sensor, but future versions could enter mass production.'
    r=analyse(candidate(text),'2026-09-20')
    assert r['flags']['early_stage'] and not r['flags']['mass_adoption']
    assert not r['flags']['limited_adoption']
    for e in r['claims']:assert text[e['start']:e['end']]==e['quote']


def test_future_demonstration_stays_future():
    r=analyse(candidate('The researchers will demonstrate a quantum sensor.'),'2026-09-20')
    assert not r['flags']['early_stage']


def test_navigation_topics_are_not_evidence():
    r=analyse(candidate('Related Topics quantum sensor prototype limited to laboratory use.'),'2026-09-20')
    assert r['claims']==[]


def test_computing_inflections_match_title_and_sentence():
    c=candidate('Researchers created a photonic computing device.',title_original='New device can compute with light')
    c['query']='photonic computing'
    r=analyse(c,'2026-09-20')
    assert r['flags']['early_stage'] and r['flags']['relevance']


def test_negated_limitation_not_positive():
    result=analyse(candidate('This quantum sensor prototype is not limited to laboratory experiments.'),'2026-09-20')
    assert not result['flags']['limited_adoption']


def test_explicit_maturity_and_conflict():
    c=candidate('This quantum sensor is widely deployed in industry.')
    assert analyse(c,'2026-09-20')['decision']=='likely_mature'
    c['sources'][0]['text']+=' This quantum sensor prototype was tested.'
    assert analyse(c,'2026-09-20')['reason']=='conflicting_stage_claims'


@pytest.mark.parametrize('changes',[
    {'published_at':None},{'published_at':'2027-01-01'},
    {'trust_level':'low'},{'source_type':'unknown'},
    {'retrieval_metadata':{'is_generated_summary':True}},
    {'retrieval_metadata':{'raw_metadata':{'text_scope':'search_snippet'}}},
])
def test_unsupported_source_abstains(changes):
    r=analyse(candidate('This quantum sensor prototype is limited to laboratory experiments.',**changes),'2026-09-20')
    assert r['decision']=='needs_review'
    assert not r['claims']


def test_russian_rules_and_exact_quotes():
    c=candidate('Наш квантовый сенсор — прототип, испытанный только в лаборатории.')
    c['query']='квантовый сенсор'
    c['sources'][0]['title_original']='Квантовый сенсор: прототип'
    assert analyse(c,'2026-09-20')['decision']=='likely_weak'


def test_quantum_ai_navigation_not_equivalent_to_quantum_sensors():
    c=candidate('We propose quantum AI for navigation using conventional sensors.')
    c['sources'][0]['title_original']='Quantum Artificial Intelligence for Vehicle Navigation'
    assert not analyse(c,'2026-09-20','quantum navigation sensors')['flags']['relevance']
