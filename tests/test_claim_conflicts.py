from src.signal_analysis import evaluate


def assessment(early, mature, feature='early_stage'):
    sources=[{'source_id':sid,'text':text,'published_at':'2026-01-01'}
             for sid,text in [('a',early),('b',mature)]]
    claims=[{'source_id':'a','quote':early,'feature':feature},
            {'source_id':'b','quote':mature,'feature':'mass_adoption'}]
    return evaluate({'sources':sources},{'claims':claims},'2026-09-27')


def test_distinct_technologies_do_not_create_established_conflict():
    r=assessment('Quantum navigation prototype is experimental.', 'Optical computing has mass adoption.')
    assert r['stage']=='uncertain'
    assert r['stage_comparisons'][0]['status']=='scope_unresolved'


def test_same_subject_is_only_a_potential_conflict():
    r=assessment('Quantum navigation prototype is experimental.', 'Quantum navigation has mass adoption.')
    assert r['stage']=='conflicting'
    assert r['stage_comparisons'][0]['status']=='potential_conflict'
    assert r['disinformation']=='not_established'


def test_limited_adoption_also_requires_comparison_with_maturity():
    r=assessment('Quantum navigation is limited to laboratory use.', 'Quantum navigation has mass adoption.', 'limited_adoption')
    assert r['stage']=='conflicting'
