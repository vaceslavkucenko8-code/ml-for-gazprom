from src.automatic import analyse
from test_automatic import candidate


def test_specific_query_term_can_be_in_sentence_not_headline():
    c = candidate('This quantum sensor prototype is limited to laboratory experiments.', title_original='A new sensor design')
    assert analyse(c, '2026-09-20')['decision'] == 'likely_weak'


def test_headline_overlap_alone_is_insufficient():
    c = candidate('Our actuator prototype is limited to laboratory experiments.', title_original='A new sensor design')
    r = analyse(c, '2026-09-20')
    assert not r['flags']['relevance']
    assert r['decision'] == 'needs_review'
