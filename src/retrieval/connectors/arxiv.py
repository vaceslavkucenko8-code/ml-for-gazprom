"""Коннектор arXiv.org — официальное открытое API научных препринтов.

Условия доступа (проверено, см. docs/source_policy.md):
  * официальный, бесплатный, без ключа: https://info.arxiv.org/help/api/index.html
  * лимит вежливости: не чаще ~1 запроса в 3 секунды на клиента — мы
    выдерживаем это через DomainThrottle в fetch.py плюс собственный запас.
  * данные из API можно использовать для некоммерческого исследовательского
    инструментария; требуется указывать arXiv как источник (см.
    https://info.arxiv.org/help/api/tou.html) — это отражено в
    source_type=ACADEMIC + сохранении оригинальной ссылки на arxiv.org.

Формат ответа — Atom XML. Мы получаем сразу заголовок, авторов, аннотацию
(abstract) и дату публикации — этого достаточно, чтобы сформировать
SourceDocument без дополнительного HTTP-запроса на HTML-страницу.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree as ET

from ..fetch import Fetcher, get_default_fetcher
from .base import ConnectorError, SearchConnector, SearchHit

logger = logging.getLogger(__name__)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _text(el: Optional[ET.Element]) -> Optional[str]:
    if el is None or el.text is None:
        return None
    return " ".join(el.text.split())


class ArxivConnector(SearchConnector):
    name = "arxiv"
    description = "arXiv.org — открытый архив научных препринтов (физика, CS, математика и др.)"
    terms_url = "https://info.arxiv.org/help/api/tou.html"
    default_source_type = "academic_paper"
    requires_api_key = False

    def __init__(self, fetcher: Optional[Fetcher] = None):
        self.fetcher = fetcher or get_default_fetcher()

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        # arXiv допускает поиск по all: / ti: / abs: — используем all: для
        # максимального охвата свободной темы.
        # A field prefix on the whole unquoted string can produce broad OR results.
        import re
        tokens = re.findall(r"[\w-]+", query)
        search_query = ' AND '.join(f'all:"{token}"' for token in tokens)
        if self.publication_window:
            start, end = (datetime.fromisoformat(d).strftime('%Y%m%d') for d in self.publication_window)
            search_query = f'({search_query}) AND submittedDate:[{start}0000 TO {end}2359]'
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        result = self.fetcher.get(ARXIV_API_URL, params=params)
        if result.status != result.status.OK or not result.content:
            raise ConnectorError(f"arXiv недоступен: {result.status.value} {result.error or ''}")

        try:
            root = ET.fromstring(result.content)
        except ET.ParseError as exc:
            raise ConnectorError(f"arXiv: не удалось разобрать Atom-ответ: {exc}") from exc

        hits: list[SearchHit] = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            url = _text(entry.find(f"{ATOM_NS}id"))
            title = _text(entry.find(f"{ATOM_NS}title"))
            summary = _text(entry.find(f"{ATOM_NS}summary"))
            published_raw = _text(entry.find(f"{ATOM_NS}published"))
            published_at = None
            if published_raw:
                try:
                    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None

            authors = [
                _text(a.find(f"{ATOM_NS}name"))
                for a in entry.findall(f"{ATOM_NS}author")
            ]
            authors = [a for a in authors if a]

            if not url:
                continue

            hits.append(
                SearchHit(
                    url=url,
                    title=title,
                    snippet=summary,
                    published_at=published_at,
                    connector=self.name,
                    query=query,
                    full_text=summary,  # аннотация — это и есть основной текст для наших целей
                    language_hint="en",  # подавляющее большинство arXiv — на английском
                    raw={"authors": authors},
                )
            )
        return hits
