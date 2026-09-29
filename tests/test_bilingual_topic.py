from src.automatic import analyse
from test_automatic import candidate
from src.topic_terms import search_equivalent


def test_query_conversion_is_complete_or_keeps_user_text():
    assert search_equivalent('Квантовые сенсоры для навигации') == 'quantum sensor navigation'
    assert search_equivalent('Квантовые сенсоры для неизвестной задачи') == 'Квантовые сенсоры для неизвестной задачи'
    assert search_equivalent('quantum sensor') == 'quantum sensor'


def test_russian_query_matches_original_english_evidence():
    c = candidate('This quantum sensor prototype is limited to laboratory experiments.')
    c['query'] = 'Квантовые сенсоры'
    result = analyse(c, '2026-09-20')
    assert result['decision'] == 'likely_weak'
    assert all(c['sources'][0]['text'][e['start']:e['end']] == e['quote'] for e in result['claims'])


def test_translation_does_not_supply_absent_stage():
    c = candidate('This quantum sensor measures magnetic fields.')
    c['query'] = 'Квантовые сенсоры'
    result = analyse(c, '2026-09-20')
    assert result['flags']['relevance']
    assert not result['flags']['limited_adoption']
    assert result['decision'] == 'needs_review'


def test_bilingual_unrelated_document_stays_unrelated():
    c = candidate('Our robotic actuator prototype is limited to laboratory experiments.', title_original='Robotic actuator')
    c['query'] = 'Квантовые сенсоры'
    assert not analyse(c, '2026-09-20')['flags']['early_stage']
