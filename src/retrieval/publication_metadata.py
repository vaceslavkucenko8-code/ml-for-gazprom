"""Recover arXiv dates from the same version's abstract page, never its ID."""
import re
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from .parse import _extract_html_meta, _extract_published_at


def recover_arxiv_date(doc, fetcher):
    if doc.published_at is not None:
        return doc
    parsed = urlsplit(doc.url)
    if parsed.hostname not in ('arxiv.org', 'www.arxiv.org'):
        return doc
    match = re.fullmatch(r'/(?:html|abs)/(\d{4}\.\d{4,5}(?:v\d+)?)', parsed.path)
    if not match:
        return doc
    url = 'https://arxiv.org/abs/' + match[1]
    response = fetcher.get(url)
    if response.status.value != 'ok' or not response.content:
        return doc
    soup = BeautifulSoup(response.content, 'html.parser')
    meta = _extract_html_meta(soup)
    normalize = lambda s: re.sub(r'\W+', '', s.casefold())
    if not meta.get('citation_title') or normalize(meta['citation_title']) != normalize(doc.title or ''):
        return doc
    published, estimated = _extract_published_at(soup, meta)
    if published is not None:
        doc.published_at = published
        doc.published_at_is_estimated = estimated
        doc.raw_metadata['publication_date_recovery'] = {
            'url': url, 'method': 'same_arxiv_id_and_title_publisher_metadata',
            'metadata': {k: v for k, v in meta.items() if k.startswith('citation_')},
        }
    return doc
