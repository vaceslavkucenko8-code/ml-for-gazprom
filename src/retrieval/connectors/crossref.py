"""Коннектор Crossref — официальный бесплатный REST API метаданных научных
публикаций (журнальные статьи, отчёты, материалы конференций).

Условия доступа (проверено, см. docs/source_policy.md):
  * официальный, бесплатный, без обязательной регистрации:
    https://www.crossref.org/documentation/retrieve-metadata/rest-api/
  * рекомендуется входить в "polite pool" — указывать контактный email в
    User-Agent или параметре mailto для более высокого лимита и приоритета
    (https://github.com/CrossRef/rest-api-doc#etiquette). Реализовано ниже
    через параметр mailto.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from ..fetch import Fetcher, get_default_fetcher
from .base import ConnectorError, SearchConnector, SearchHit

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"
POLITE_EMAIL = os.environ.get("CROSSREF_POLITE_EMAIL")


def _parse_date_parts(date_parts_obj: Optional[dict]) -> Optional[datetime]:
    if not date_parts_obj:
        return None
    parts = date_parts_obj.get("date-parts")
    if not parts or not parts[0]:
        return None
    p = parts[0] + [1, 1]  # добиваем месяц/день по умолчанию
    try:
        return datetime(p[0], p[1], p[2], tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class CrossrefConnector(SearchConnector):
    name = "crossref"
    description = "Crossref — реестр метаданных научных публикаций (DOI)"
    terms_url = "https://www.crossref.org/documentation/retrieve-metadata/rest-api/"
    default_source_type = "academic_paper"
    requires_api_key = False

    def __init__(self, fetcher: Optional[Fetcher] = None):
        self.fetcher = fetcher or get_default_fetcher()

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        params = {
            "query": query,
            "rows": max_results,
            "select": "title,DOI,URL,abstract,published,container-title,author,type",
        }
        if POLITE_EMAIL:
            params['mailto'] = POLITE_EMAIL
        if self.publication_window:
            start, end = self.publication_window
            params['filter'] = f'from-pub-date:{start},until-pub-date:{end}'
        result = self.fetcher.get(CROSSREF_API_URL, params=params)
        if result.status != result.status.OK or not result.content:
            raise ConnectorError(f"Crossref недоступен: {result.status.value} {result.error or ''}")

        import json

        try:
            data = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"Crossref: не удалось разобрать JSON: {exc}") from exc

        items = data.get("message", {}).get("items", [])
        hits: list[SearchHit] = []
        for item in items:
            titles = item.get("title") or []
            title = titles[0] if titles else None
            url = item.get("URL")
            if not url:
                continue
            abstract = item.get("abstract")
            published_at = _parse_date_parts(item.get("published"))
            authors = [
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in (item.get("author") or [])
            ]
            hits.append(
                SearchHit(
                    url=url,
                    title=title,
                    snippet=abstract,
                    published_at=published_at,
                    connector=self.name,
                    query=query,
                    full_text=abstract,
                    language_hint=item.get("language"),
                    raw={
                        "doi": item.get("DOI"),
                        "container_title": item.get("container-title"),
                        "type": item.get("type"),
                        "authors": authors,
                        "date_estimated": len((item.get('published') or {}).get('date-parts', [[]])[0]) < 3,
                    },
                )
            )
        return hits
