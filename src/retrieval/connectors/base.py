"""Базовый интерфейс коннектора поиска.

Каждый коннектор оборачивает один открытый источник/API. Контракт простой и
намеренно узкий: коннектор получает поисковый запрос и возвращает список
"сырых находок" (SearchHit) — ссылка + минимальные метаданные, без полного
текста. Полный текст и остальные метаданные достраиваются позже в
fetch.py/parse.py (для API, которые уже отдают текст/аннотацию — как arXiv
или Crossref, — parse.py использует то, что уже есть, не делая лишний HTTP-запрос).

Это разделение соответствует архитектурному требованию ТЗ:
"с четким разделением логики парсинга, инференса модели и интерфейса" —
поиск (кто и где ищет), загрузка (как забрать контент) и разбор (как
превратить контент в SourceDocument) — три независимых, тестируемых слоя.
"""

from __future__ import annotations

import abc
import dataclasses
from datetime import datetime
from typing import Any, Optional


@dataclasses.dataclass
class SearchHit:
    """Единичный результат поиска до загрузки/разбора."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    published_at: Optional[datetime] = None
    connector: str = "unknown"
    query: str = ""
    # Если коннектор — это структурированное API (не веб-страница), он может
    # сразу отдать текст/аннотацию, язык и т.п. — тогда parse.py не будет
    # делать лишний HTTP fetch.
    full_text: Optional[str] = None
    language_hint: Optional[str] = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)


class ConnectorError(RuntimeError):
    """Ошибка коннектора, которая не должна прерывать весь сбор (см. п. 2.2:
    "ошибка одной страницы не прекращает весь сбор"). Оркестратор в search.py
    ловит это исключение на уровне отдельного коннектора/запроса.
    """


class SearchConnector(abc.ABC):
    # Optional ISO-date window applied by APIs supporting publication filters.
    # Other connectors still rely on the downstream date check.
    publication_window: Optional[tuple[str, str]] = None
    #: машинное имя коннектора, используется в SourceDocument.connector
    name: str = "base"
    #: человекочитаемое описание источника — для docs/source_policy.md и логов
    description: str = ""
    #: URL официальной документации / условий использования API
    terms_url: str = ""
    #: базовый тип источника, который даёт этот коннектор (используется как
    #: подсказка для parse.py/deduplicate.py, может быть переопределён по
    #: содержимому конкретного документа)
    default_source_type: str = "unknown"
    #: нужен ли API-ключ; если True и ключ не задан — коннектор должен молча
    #: отключаться (is_available() -> False), а не падать
    requires_api_key: bool = False

    def is_available(self) -> bool:
        return True

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        """Выполнить поиск. Обязан бросать только ConnectorError на ожидаемые
        сбои (таймаут, недоступность, отсутствие ключа) — сетевые/HTTP детали
        оборачиваются внутри коннектора.
        """
        raise NotImplementedError
