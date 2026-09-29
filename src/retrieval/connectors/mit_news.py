"""Public institutional news search; results must be fetched as original pages."""
import re
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup
from .base import SearchConnector, SearchHit, ConnectorError
from ..fetch import get_default_fetcher


class MitNewsConnector(SearchConnector):
    name = 'mit_news'
    description = 'MIT News — institutional research news, not peer review'
    terms_url = 'https://news.mit.edu/terms-use'
    default_source_type = 'university'

    def __init__(self, fetcher=None):
        self.fetcher = fetcher or get_default_fetcher()

    def search(self, query, max_results=10):
        result = self.fetcher.get('https://news.mit.edu/search', params={'keyword': query})
        if result.status.value != 'ok' or not result.content:
            raise ConnectorError(f'MIT News search unavailable: {result.status.value}')
        soup = BeautifulSoup(result.content, 'html.parser')
        soups = [soup]
        visited = set()
        warnings = []
        # Follow the publisher's actual pagination links, at most three pages.
        # Summaries remain discovery data and are never treated as evidence.
        while sum(len(s.select('article.search-result-item')) for s in soups) < max_results and len(soups) < 3:
            next_link = soups[-1].select_one('a[rel~=next][href]')
            if not next_link:
                break
            next_url = urljoin('https://news.mit.edu', next_link['href'])
            parsed = urlsplit(next_url)
            if parsed.scheme != 'https' or parsed.hostname != 'news.mit.edu' or parsed.path != '/search' or next_url in visited:
                break
            visited.add(next_url)
            page = self.fetcher.get(next_url)
            if page.status.value != 'ok' or not page.content:
                warnings.append(f'MIT pagination unavailable: {page.status.value}')
                break
            soups.append(BeautifulSoup(page.content, 'html.parser'))
        # The publisher returns a chronological list. Rank the visible result
        # summaries before applying the download limit, never use them as evidence.
        def terms(text):
            from ...automatic import words
            return {re.sub(r'^comput(?:e|es|er|ers|ing|ation|ational)$', 'comput', t) for t in words(text)}
        topic = terms(query)
        results = [item for page in soups for item in page.select('article.search-result-item')]
        if results:
            ranked = []
            for index, item in enumerate(results):
                link = item.select_one('.search-result-item--title a[href]')
                if link:
                    overlap = len(topic & terms(item.get_text(' ', strip=True)))
                    ranked.append((-overlap, index, link))
            links = [link for _, _, link in sorted(ranked, key=lambda x: x[:2])]
        else:
            links = soup.select('a[href]')
        hits, seen = [], set()
        for link in links:
            url = urljoin('https://news.mit.edu', link['href'])
            parsed = urlsplit(url)
            title = link.get_text(' ', strip=True)
            if parsed.hostname != 'news.mit.edu' or not re.match(r'^/\d{4}/[^/]+$', parsed.path):
                continue
            if not title or url in seen:
                continue
            seen.add(url)
            hits.append(SearchHit(url=url, title=title, connector=self.name, query=query,
                                  language_hint='en', raw={'discovery': 'institutional_search',
                                                          'search_pages': len(soups), 'discovery_warnings': warnings}))
            if len(hits) >= max_results:
                break
        return hits
