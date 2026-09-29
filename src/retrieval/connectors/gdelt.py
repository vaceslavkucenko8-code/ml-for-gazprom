"""Коннектор GDELT DOC 2.0 API — открытая база мировых новостных публикаций.

Условия доступа (проверено, см. docs/source_policy.md):
  * официальный, полностью бесплатный, без ключа и без ограничения по
    некоммерческому/коммерческому использованию:
    https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    https://github.com/CIA-Consulting/gdelt-doc-api (неофициальная обёртка;
    мы используем эндпоинт напрямую, без сторонних зависимостей).
  * покрывает десятки языков, включая русский — это наш основной канал для
    расширения русскоязычного покрытия (п. 2.5 ТЗ) без необходимости
    отдельного платного API.
  * это единственный источник "общих новостей/отраслевых медиа" в наборе —
    поэтому по умолчанию мы помечаем находки как GENERAL_MEDIA/INDUSTRY_MEDIA
    (уточняется эвристикой по домену в parse.py), а не как первоисточник.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from ..fetch import Fetcher, get_default_fetcher
from .base import ConnectorError, SearchConnector, SearchHit

logger = logging.getLogger(__name__)

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_gdelt_date(raw: Optional[str]) -> Optional[datetime]:
    # GDELT отдаёт seendate в формате YYYYMMDDHHMMSS
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class GdeltConnector(SearchConnector):
    name = "gdelt"
    description = "GDELT DOC 2.0 — открытая многоязычная база новостных публикаций"
    terms_url = "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/"
    default_source_type = "general_media"
    requires_api_key = False

    def __init__(self, fetcher: Optional[Fetcher] = None, language: Optional[str] = None):
        self.fetcher = fetcher or get_default_fetcher()
        self.language = language  # напр. "rus" чтобы ограничить язык результатов

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        q = query
        if self.language:
            q = f"{q} sourcelang:{self.language}"
        params = {
            "query": q,
            "mode": "artlist",
            "maxrecords": min(max_results, 250),
            "format": "json",
            "sort": "hybridrel",
        }
        result = self.fetcher.get(GDELT_API_URL, params=params)
        if result.status != result.status.OK or not result.content:
            raise ConnectorError(f"GDELT недоступен: {result.status.value} {result.error or ''}")

        try:
            data = json.loads(result.content)
        except json.JSONDecodeError as exc:
            # GDELT иногда возвращает пустой ответ/HTML при перегрузке — не валим весь сбор
            raise ConnectorError(f"GDELT: не удалось разобрать JSON: {exc}") from exc

        articles = data.get("articles", []) or []
        hits: list[SearchHit] = []
        for art in articles[:max_results]:
            url = art.get("url")
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=art.get("title"),
                    snippet=None,  # GDELT artlist не отдаёт полный текст — дозагрузим в fetch.py
                    published_at=_parse_gdelt_date(art.get("seendate")),
                    connector=self.name,
                    query=query,
                    language_hint=art.get("language"),
                    raw={
                        "domain": art.get("domain"),
                        "source_country": art.get("sourcecountry"),
                        "social_image": art.get("socialimage"),
                    },
                )
            )
        return hits
