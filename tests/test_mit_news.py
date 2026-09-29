from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.retrieval.connectors.mit_news import MitNewsConnector
from src.retrieval.connectors.base import ConnectorError


def test_discovery_keeps_original_page_and_no_invented_date():
    fetcher=Mock()
    fetcher.get.return_value=SimpleNamespace(status=SimpleNamespace(value='ok'),content='''
      <a href="/2025/new-sensor-1117">New sensor</a>
      <a href="/2025/new-sensor-1117">Duplicate</a>
      <a href="https://news.mit.edu.evil.example/2025/fake">Bad</a>
      <a href="/topics/sensors">Navigation</a>''')
    hits=MitNewsConnector(fetcher).search('sensor')
    assert len(hits)==1 and hits[0].url=='https://news.mit.edu/2025/new-sensor-1117'
    assert hits[0].full_text is None and hits[0].published_at is None
    assert fetcher.get.call_args.kwargs['params']=={'keyword':'sensor'}


def test_search_failure_is_reported():
    fetcher=Mock()
    fetcher.get.return_value=SimpleNamespace(status=SimpleNamespace(value='blocked'),content='')
    with pytest.raises(ConnectorError):MitNewsConnector(fetcher).search('sensor')


def test_topic_ranking_precedes_download_limit():
    fetcher=Mock()
    fetcher.get.return_value=SimpleNamespace(status=SimpleNamespace(value='ok'),content='''
    <article class="search-result-item"><h3 class="search-result-item--title"><a href="/2026/chip">Photonic chip</a></h3><p>Wearable displays</p></article>
    <article class="search-result-item"><h3 class="search-result-item--title"><a href="/2026/compute">Compute with light</a></h3><p>Photonic devices</p></article>''')
    hits=MitNewsConnector(fetcher).search('photonic computing',max_results=1)
    assert hits[0].url.endswith('/compute')
    assert hits[0].full_text is None


def test_next_page_followed_and_deduplicated():
    fetcher = Mock()
    def page(content):
        return SimpleNamespace(status=SimpleNamespace(value='ok'), content=content)
    article = '<article class="search-result-item"><h3 class="search-result-item--title"><a href="/2026/a">Photonic sensor</a></h3></article>'
    fetcher.get.side_effect = [page(article + '<a rel="next" href="/search?keyword=photonic&page=1">Next page</a>'),
                               page(article + article.replace('/2026/a', '/2026/b'))]
    hits = MitNewsConnector(fetcher).search('photonic', max_results=2)
    assert [h.url for h in hits] == ['https://news.mit.edu/2026/a', 'https://news.mit.edu/2026/b']
    assert all(h.raw['search_pages'] == 2 for h in hits)
    assert fetcher.get.call_args.args == ('https://news.mit.edu/search?keyword=photonic&page=1',)


def test_unavailable_next_page_keeps_first_page():
    fetcher = Mock()
    fetcher.get.side_effect = [SimpleNamespace(status=SimpleNamespace(value='ok'), content='''
      <article class="search-result-item"><h3 class="search-result-item--title"><a href="/2026/a">Sensor</a></h3></article>
      <a rel="next" href="/search?page=1">Next</a>'''),
      SimpleNamespace(status=SimpleNamespace(value='blocked'), content='')]
    hits = MitNewsConnector(fetcher).search('sensor', max_results=2)
    assert len(hits) == 1 and hits[0].raw['discovery_warnings']
