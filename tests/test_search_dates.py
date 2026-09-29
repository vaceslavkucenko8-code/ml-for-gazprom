from unittest.mock import Mock
from types import SimpleNamespace
from src.retrieval.schemas import FetchStatus
from src.retrieval.connectors.arxiv import ArxivConnector
from src.retrieval.connectors.crossref import CrossrefConnector


def test_arxiv_window_keeps_topic_conjunction():
    fetcher = Mock()
    fetcher.get.return_value = SimpleNamespace(status=FetchStatus.OK, content='<feed xmlns="http://www.w3.org/2005/Atom"/>')
    connector = ArxivConnector(fetcher)
    connector.publication_window = ('2025-04-06', '2026-09-28')
    assert connector.search('photonic computing') == []
    query = fetcher.get.call_args.kwargs['params']['search_query']
    assert query == '(all:"photonic" AND all:"computing") AND submittedDate:[202504060000 TO 202609282359]'


def test_crossref_window_and_unconfigured_compatibility():
    fetcher = Mock()
    fetcher.get.return_value = SimpleNamespace(status=FetchStatus.OK, content='{"message":{"items":[]}}')
    connector = CrossrefConnector(fetcher)
    connector.search('photonic computing')
    assert 'filter' not in fetcher.get.call_args.kwargs['params']
    connector.publication_window = ('2025-04-06', '2026-09-28')
    connector.search('photonic computing')
    assert fetcher.get.call_args.kwargs['params']['filter'] == 'from-pub-date:2025-04-06,until-pub-date:2026-09-28'
