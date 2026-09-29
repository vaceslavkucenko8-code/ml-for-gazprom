from src.retrieval.parse import guess_source_type
from src.retrieval.schemas import SourceType
from src.retrieval.schemas import SourceDocument, FetchStatus, TrustLevel
from src.retrieval.deduplicate import deduplicate_and_score
from unittest.mock import patch


def test_heise_editorial_section_only():
    assert guess_source_type('https://www.heise.de/en/news/example.html')[0] == SourceType.INDUSTRY_MEDIA
    assert guess_source_type('https://www.heise.de/shopping/example')[0] != SourceType.INDUSTRY_MEDIA
    assert guess_source_type('https://heise.de.evil.example/en/news/example')[0] != SourceType.INDUSTRY_MEDIA


def test_distinct_domains_do_not_prove_independent_confirmation():
    docs = [SourceDocument(doc_id=str(i), url=f'https://mirror{i}.example/news',
             normalized_url=f'https://mirror{i}.example/news',
             title='A company announcement', text=f'Announcement edition {i}',
             source_type=SourceType.PRESS_RELEASE, source_type_confidence=.9,
             fetch_status=FetchStatus.OK, event_cluster_id='same') for i in range(2)]
    with patch('src.retrieval.deduplicate.annotate_duplicates', side_effect=lambda d: d), patch('src.retrieval.deduplicate.cluster_by_event', side_effect=lambda d: d):
        result = deduplicate_and_score(docs)
    assert all(d.trust_level == TrustLevel.LOW for d in result)
