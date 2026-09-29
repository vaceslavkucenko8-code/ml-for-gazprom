"""Коннектор GitHub REST API (поиск репозиториев).

Условия доступа (проверено):
  * официальный API, бесплатный; без токена — 10 запросов/мин на IP, с
    персональным токеном (env GITHUB_TOKEN) — 30/мин. Мы работаем и без
    токена (requires_api_key=False), но подхватываем токен, если он задан.
    https://docs.github.com/en/rest/search
  * сигнал разработческой активности вокруг новой технологии (свежие
    репозитории, быстрый рост звёзд) — хороший ранний индикатор, source_type
    помечается как CODE_REPOSITORY и по умолчанию не является самостоятельным
    основанием для кандидата без независимого подтверждения.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

from ..fetch import Fetcher, get_default_fetcher
from .base import ConnectorError, SearchConnector, SearchHit

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com/search/repositories"


class GitHubConnector(SearchConnector):
    name = "github"
    description = "GitHub REST API — поиск репозиториев (сигнал разработческой активности)"
    terms_url = "https://docs.github.com/en/rest/search"
    default_source_type = "code_repository"
    requires_api_key = False  # работает и без токена, но с более низким лимитом

    def __init__(self, fetcher: Optional[Fetcher] = None, token: Optional[str] = None):
        self.fetcher = fetcher or get_default_fetcher()
        self.token = token or os.environ.get("GITHUB_TOKEN")

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": max_results,
        }
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        result = self.fetcher.get(GITHUB_API_URL, params=params, headers=headers)
        if result.status != result.status.OK or not result.content:
            raise ConnectorError(f"GitHub недоступен: {result.status.value} {result.error or ''}")

        try:
            data = json.loads(result.content)
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"GitHub: не удалось разобрать JSON: {exc}") from exc

        hits: list[SearchHit] = []
        for repo in data.get("items", []):
            url = repo.get("html_url")
            if not url:
                continue
            created_at = repo.get("created_at")
            published_at = None
            if created_at:
                try:
                    published_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    pass
            hits.append(
                SearchHit(
                    url=url,
                    title=repo.get("full_name"),
                    snippet=repo.get("description"),
                    published_at=published_at,
                    connector=self.name,
                    query=query,
                    full_text=repo.get("description") or "",
                    language_hint="en",
                    raw={
                        "stars": repo.get("stargazers_count"),
                        "language": repo.get("language"),
                        "topics": repo.get("topics"),
                        "pushed_at": repo.get("pushed_at"),
                    },
                )
            )
        return hits
