import copy
from types import SimpleNamespace
from src.retrieval.enrichment import targets, enrich_payload
from src.extraction.candidates import cluster_documents_into_candidates
from src.retrieval.schemas import FetchStatus
from test_pipeline import doc


def test_targets_preserve_version_and_reject_non_scholarly_links():
    source = doc(url='https://arxiv.org/abs/2602.12345v2', raw_metadata={'research_links': [
        'https://doi.org/10.1234/paper', 'http://127.0.0.1/secret',
        'https://doi.org.evil.example/10.1234/fake']})
    assert list(targets(source)) == [
        ('https://arxiv.org/html/2602.12345v2', 'same_work_full_text'),
        ('https://doi.org/10.1234/paper', 'cited_work')]


def test_identity_merges_abstract_and_full_text_but_not_citations():
    abstract = doc(doc_id='a', url='https://arxiv.org/abs/2602.12345v1', title='Photonic sensor')
    full = doc(doc_id='b', url='https://arxiv.org/html/2602.12345v1', title='Unrelated display title')
    citing = doc(doc_id='c', url='https://example.org/news', title='Other technology',
                 raw_metadata={'research_links': [abstract.url]})
    groups = cluster_documents_into_candidates([abstract, full, citing])
    assert [[d.doc_id for d in g] for g in groups] == [['a', 'b'], ['c']]


def test_failed_enrichment_preserves_input_and_does_not_invent_evidence():
    source = doc(url='https://arxiv.org/abs/2602.12345v1')
    payload = {'query': 'quantum navigation', 'documents': [source.model_dump(mode='json')]}
    original = copy.deepcopy(payload)
    fetcher = SimpleNamespace(get=lambda url: SimpleNamespace(url=url, final_url=url,
        status=FetchStatus.BLOCKED, status_code=403, content=None, error='HTTP 403'))
    enriched = enrich_payload(payload, '2026-09-28', fetcher)
    assert payload == original
    assert enriched['documents'] == payload['documents']
    assert enriched['summary']['enrichment']['attempts'][0]['status'] == 'blocked'
    assert enriched['summary']['enrichment']['fetched'] == 0
    assert enriched['summary']['errors'][0]['connector'] == 'scholarly_followup'


def test_enrichment_full_text_is_not_independent_and_has_own_date():
    source = doc(url='https://arxiv.org/abs/2602.12345v1')
    payload = {'query': 'quantum navigation', 'documents': [source.model_dump(mode='json')]}
    html = '''<html><head><title>Quantum navigation sensor</title>
    <meta name="citation_publication_date" content="2026-09-02"></head><body>
    <article><p>Our quantum navigation prototype is limited to laboratory tests.</p></article></body></html>'''
    fetcher = SimpleNamespace(get=lambda url: SimpleNamespace(url=url, final_url=url,
        status=FetchStatus.OK, status_code=200, content=html, content_type='text/html'))
    enriched = enrich_payload(payload, '2026-09-28', fetcher)
    assert len(enriched['documents']) == 2
    assert enriched['summary']['enrichment']['fetched'] == 1
    assert enriched['summary']['enrichment']['independence_verified'] is False
    extra = enriched['documents'][-1]
    assert extra['published_at'].startswith('2026-09-02')
    assert extra['raw_metadata']['discovered_from'] == source.url


def test_enrichment_obeys_age_limit_and_query_override():
    source = doc(url='https://arxiv.org/abs/2602.12345v1')
    payload = {'query': 'quantum navigation', 'documents': [source.model_dump(mode='json')]}
    def unexpected(*args):
        raise AssertionError('Filtered source should not trigger enrichment')
    fetcher = SimpleNamespace(get=unexpected)
    for settings in ({'max_age_days': 1}, {'search_query': 'battery chemistry'}):
        result = enrich_payload(payload, '2026-09-28', fetcher, **settings)
        assert result['summary']['enrichment']['attempts'] == []
