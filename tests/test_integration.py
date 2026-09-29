import copy
import json
from pathlib import Path
import pytest
from src.participant2_adapter import adapt_export
from src.detector import assess
from src.extraction.candidates import split_sentences
from src.extraction.features import candidate_features
from src.retrieval.schemas import Candidate, SourceDocument, Evidence, TrustLevel


def payload():
    return {'documents': [{'doc_id': 'd', 'url': 'https://example.org/pilot',
        'title': 'Pilot', 'text': 'A prototype was tested in the lab.',
        'source_type': 'academic_paper', 'trust_level': 'high',
        'trust_explanation': 'Test fixture', 'published_at': '2026-09-01T00:00:00Z'}],
        'candidates': [{'candidate_id': 'c', 'technology_name': 'Сенсор', 'query': 'сенсоры',
        'source_doc_ids': ['d'], 'evidence': [{'evidence_id': 'e', 'doc_id': 'd',
        'quote': 'A prototype was tested in the lab.'}]}], 'synthetic': True}


def test_no_automatic_semantic_review():
    adapted = adapt_export(payload())
    c = adapted['candidates'][0]
    assert not c['evidence']
    assert len(c['extracted_evidence_pending']) == 1
    assert assess(c, '2026-09-20')['decision'] == 'needs_review'


def test_review_is_explicit_and_preserves_traceability():
    r = {'candidate_id': 'c', 'evidence_id': 'e', 'feature': 'early_stage',
         'reviewer': 'tester', 'rationale': 'Synthetic test', 'reviewed': True,
         'scope_matches_candidate': True, 'event_status': 'observed', 'observed_at': '2026-09-01'}
    c = adapt_export(payload(), [r])['candidates'][0]
    assert assess(c, '2026-09-20')['accepted_evidence_ids'] == ['e:early_stage']
    bad = payload()
    bad['candidates'][0]['evidence'][0]['quote'] = 'invented'
    with pytest.raises(ValueError, match='Quote'):
        adapt_export(bad, [r])
    bad = payload()
    bad['candidates'][0]['evidence'][0]['is_future_looking'] = True
    with pytest.raises(ValueError, match='Future'):
        adapt_export(bad, [r])


def test_duplicate_document_rejected():
    p = payload()
    p['documents'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        adapt_export(p)


def test_quote_whitespace_preserved():
    text = 'A prototype\n was tested in a laboratory. Another  sentence about a pilot.'
    assert all(q in text for q in split_sentences(text))


def test_unrelated_document_cannot_change_candidate_features():
    c = Candidate(technology_name='test', short_description='test', query='test', source_doc_ids=['missing'])
    d = SourceDocument(doc_id='other', url='https://example.org', normalized_url='https://example.org',
                       title='other', text='other', connector='test', query='test', trust_level=TrustLevel.HIGH)
    assert candidate_features(c, [d])['high_trust_source_count'] == 0


def test_quotes_not_counted_as_separate_sources():
    e = Evidence(doc_id='d', url='https://example.org', quote='text', trust_level=TrustLevel.HIGH)
    c = Candidate(technology_name='test', short_description='test', query='test', evidence=[e, e.model_copy()])
    assert candidate_features(c)['high_trust_source_count'] == 1


def test_review_pack_preserves_annotations(tmp_path, monkeypatch):
    from src import build_review_pack
    project = Path(__file__).resolve().parents[1]
    import shutil
    if not (project/'data/processed').exists():
        import pytest
        pytest.skip('Private processed dataset not published')
    shutil.copytree(project/'data/processed', tmp_path/'data/processed')
    (tmp_path/'artifacts').mkdir()
    shutil.copy2(project/'artifacts/oof_predictions.csv', tmp_path/'artifacts/oof_predictions.csv')
    monkeypatch.chdir(tmp_path)
    build_review_pack.run()
    target = tmp_path/'data/review/candidates_pending.json'
    value = json.loads(target.read_text(encoding='utf-8'))
    value['candidates'][0]['review_note_ru'] = 'Human annotation to retain'
    target.write_text(json.dumps(value), encoding='utf-8')
    before = target.read_bytes()
    build_review_pack.run()
    assert target.read_bytes() == before
