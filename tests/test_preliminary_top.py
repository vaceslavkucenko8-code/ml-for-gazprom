import json
from src.automatic import analyse, preliminary_candidates
from src.presentation_view import load_view
from test_automatic import candidate


def assessment(cid, score, early=True, relevance=True, decision='needs_review',
               reason='insufficient_explicit_evidence', **flags):
    return {'candidate_id': cid, 'decision': decision, 'reason': reason, 'score': score,
            'flags': {'early_stage': early, 'relevance': relevance, **flags}, 'claims': [], 'ignored': []}


def test_prototype_alone_stays_needs_review_but_is_preliminary():
    r = analyse(candidate('This quantum sensor prototype was tested in a pilot project.'), '2026-09-20')
    assert r['decision'] == 'needs_review'
    assert [a['candidate_id'] for a in preliminary_candidates([r])] == ['c']


def test_preliminary_requires_early_stage_and_relevance_without_conflicts():
    items = [assessment('ok', 60),
             assessment('no_early', 30, early=False),
             assessment('no_relevance', 30, relevance=False),
             assessment('conflict', 60, reason='conflicting_stage_claims'),
             assessment('mature', 60, mass_adoption=True),
             assessment('strict', 100, decision='likely_weak',
                        reason='explicit_early_and_limited_scope_claims')]
    assert [a['candidate_id'] for a in preliminary_candidates(items)] == ['ok']


def test_preliminary_order_limit_and_exclusions():
    items = [assessment(f'c{i}', 60 if i % 2 else 30) for i in range(6)]
    got = [a['candidate_id'] for a in preliminary_candidates(items, exclude_ids={'c1'}, limit=3)]
    assert got == ['c3', 'c5', 'c0']
    assert preliminary_candidates(items, limit=0) == []


def test_view_fills_only_free_slots_after_strict_top(tmp_path):
    folder = tmp_path / 'run'; folder.mkdir()
    source = {'source_id': 's', 'url': 'https://example.org', 'title_original': 'Sensor', 'text': 'Sensor text.',
              'source_type': 'research', 'trust_level': 'high', 'retrieval_metadata': {}}
    candidates = [{'candidate_id': f'c{i}', 'name_ru': 'Датчик', 'query': 'Датчики', 'sources': [source]}
                  for i in range(20)]
    strict = [assessment(f'c{i}', 100, decision='likely_weak',
                         reason='explicit_early_and_limited_scope_claims') for i in range(10)]
    weaker = [assessment(f'c{i}', 60) for i in range(10, 20)]
    for name, data in {'automatic.json': {'assessments': strict + weaker},
                       'detector_input.json': {'candidates': candidates},
                       'summary.json': {'query': 'Датчики', 'as_of': '2026-09-23'},
                       'result.json': {'results': strict}}.items():
        (folder / name).write_text(json.dumps(data))
    view = load_view(folder, translate_cards=False)
    assert view['top_ids'] == [f'c{i}' for i in range(10)]
    assert view['preliminary_ids'] == [f'c{i}' for i in range(10, 15)]
    assert all(c['assessment']['decision'] == 'needs_review'
               for c in view['cards'] if c['candidate_id'] in view['preliminary_ids'])
