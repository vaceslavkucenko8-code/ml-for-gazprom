"""Коннектор Hacker News (через Algolia HN Search API).

Условия доступа (проверено):
  * официальный публичный API, бесплатный, без ключа:
    https://hn.algolia.com/api
  * полезен как ранний индикатор технологического интереса сообщества
    разработчиков — часто опережает "деловые" медиа на месяцы, что хорошо
    соответствует идее "слабого сигнала". Используется как ДОПОЛНИТЕЛЬНЫЙ,
    низко-доверенный (по умолчанию) источник: обсуждения — это не
    первоисточник, но полезный триггер для дальнейшей проверки.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from ..fetch import Fetcher, get_default_fetcher
from .base import ConnectorError, SearchConnector, SearchHit

logger = logging.getLogger(__name__)

HN_API_URL = "https://hn.algolia.com/api/v1/search"


class HackerNewsConnector(SearchConnector):
    name = "hackernews"
    description = "Hacker News (Algolia Search API) — обсуждения технологического сообщества"
    terms_url = "https://hn.algolia.com/api"
    default_source_type = "forum"
    requires_api_key = False

    def __init__(self, fetcher: Optional[Fetcher] = None):
        self.fetcher = fetcher or get_default_fetcher()

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        params = {
            "query": query,
            "tags": "story",
            "hitsPerPage": max_results,
        }
        result = self.fetcher.get(HN_API_URL, params=params)
        if result.status != result.status.OK or not result.content:
            raise ConnectorError(f"Hacker News недоступен: {result.status.value} {result.error or ''}")

        try:
            data = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"Hacker News: не удалось разобрать JSON: {exc}") from exc

        hits: list[SearchHit] = []
        for hit in data.get("hits", []):
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            created_at = hit.get("created_at")
            published_at = None
            if created_at:
                try:
                    published_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    pass
            hits.append(
                SearchHit(
                    url=url,
                    title=hit.get("title"),
                    snippet=hit.get("story_text") or hit.get("_highlightResult", {}).get("title", {}).get("value"),
                    published_at=None,  # Submission date does not date the linked article.
                    connector=self.name,
                    query=query,
                    language_hint="en",
                    raw={"points": hit.get("points"), "num_comments": hit.get("num_comments"),
                         "discovered_at": created_at, "text_scope": "search_snippet"},
                )
            )
        return hits
