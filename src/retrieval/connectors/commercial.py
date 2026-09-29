"""Точка расширения: универсальный веб-поиск через коммерческое API.

Ни один из бесплатных keyless-источников (arXiv/Crossref/GDELT/HN/GitHub) не
даёт полноценного общего веб-поиска "как в браузере". Для промышленной
эксплуатации сервиса (в первую очередь — чтобы покрыть патентные базы,
аналитические отчёты и русскоязычные отраслевые медиа шире, чем GDELT)
предполагается подключение одного из коммерческих API с бесплатным пробным
тарифом, например:

  * Tavily Search API      https://docs.tavily.com/            (есть free tier)
  * Bing Web Search API    https://learn.microsoft.com/bing/search-apis/
  * Google Programmable Search  https://developers.google.com/custom-search
  * Yandex Search API      https://yandex.cloud/ru/docs/search-api/  (важно для RU-покрытия)

Мы НЕ подключаем платный ключ "по умолчанию" — в этой среде демонстрации его
нет и в ТЗ выбор облачных сервисов требует согласования (см. ТЗ, п. 3.1).
Вместо этого даём готовый, протестированный каркас коннектора: если задать
переменную окружения TAVILY_API_KEY, он включится автоматически
(requires_api_key=True -> is_available() отражает реальное наличие ключа) и
заработает без изменений кода. Если ключа нет — коннектор просто не
участвует в сборе (см. search.py: недоступные коннекторы пропускаются, а не
приводят к ошибке).
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

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilyConnector(SearchConnector):
    """Опциональный коннектор общего веб-поиска через Tavily API.

    Отключён по умолчанию (нет ключа в этой среде) — включается переменной
    окружения TAVILY_API_KEY. Использование условно бесплатного стороннего
    API согласуется с бизнес-заказчиком отдельно, как того требует ТЗ
    (п. 3.1 "Использование других моделей... допускается только после
    согласования с представителями бизнеса") — здесь тот же принцип
    применён к сторонним сервисам поиска.
    """

    name = "tavily"
    description = "Tavily Search API — универсальный веб-поиск (опционально, требует ключ)"
    terms_url = "https://docs.tavily.com/"
    default_source_type = "general_media"
    requires_api_key = True

    def __init__(self, fetcher: Optional[Fetcher] = None, api_key: Optional[str] = None):
        self.fetcher = fetcher or get_default_fetcher()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        if not self.api_key:
            raise ConnectorError("Tavily: не задан TAVILY_API_KEY, коннектор пропущен")

        # Tavily принимает POST, но наш Fetcher заточен под GET для остальных
        # коннекторов — здесь делаем прямой запрос через тот же session,
        # чтобы сохранить единые таймауты/ретраи.
        session = self.fetcher.session
        try:
            resp = session.post(
                TAVILY_API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                },
                timeout=(self.fetcher.connect_timeout, self.fetcher.read_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            raise ConnectorError(f"Tavily недоступен: {exc}") from exc

        if resp.status_code >= 400:
            raise ConnectorError(f"Tavily: HTTP {resp.status_code}")

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise ConnectorError(f"Tavily: не удалось разобрать JSON: {exc}") from exc

        hits: list[SearchHit] = []
        for item in data.get("results", []):
            hits.append(
                SearchHit(
                    url=item.get("url"),
                    title=item.get("title"),
                    snippet=item.get("content"),
                    published_at=None,
                    connector=self.name,
                    query=query,
                    full_text=item.get("content"),
                    raw={"score": item.get("score")},
                )
            )
        return hits
