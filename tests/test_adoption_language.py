import pytest
from src.automatic import analyse
from test_automatic import candidate


def swarm(text):
    c=candidate(text,title_original='Towards applied swarm robotics')
    c['query']='Роевая робототехника'
    return analyse(c,'2026-09-20')


def test_separate_research_and_market_evidence_with_context():
    text=('Swarm robotics studies groups of robots cooperating on tasks. '
          'Research in this field has predominantly relied on simulations. '
          'Swarm robotics has yet to see widespread commercial or industrial application.')
    r=swarm(text)
    assert r['decision']=='likely_weak'
    early=next(c for c in r['claims'] if c['feature']=='early_stage')
    assert early['scope_context']['quote'].startswith('Swarm robotics')
    assert all(text[c['start']:c['end']]==c['quote'] for c in r['claims'])


@pytest.mark.parametrize('text',[
    'Swarm robotics will reach widespread commercial adoption.',
    'It is not true that swarm robotics has yet to see widespread commercial application.',
    'Swarm robotics has achieved widespread commercial adoption.',
    'Swarm robotics is being evaluated in experiments.',
])
def test_no_limited_adoption_from_plan_negation_or_experiment(text):
    assert not swarm(text)['flags']['limited_adoption']


def test_unrelated_preceding_subject_does_not_supply_stage():
    r=swarm('Optical computing studies photonic chips and optical circuits. Research in this field has predominantly relied on simulations.')
    assert not r['flags']['early_stage']
