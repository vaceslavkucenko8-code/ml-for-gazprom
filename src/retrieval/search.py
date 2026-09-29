"""
src/retrieval/search.py

Центральный модуль поиска по свободному запросу (ТЗ, п. 2.1).

Пользователь вводит произвольную технологическую тему на русском или
английском языке ("технологии в ИИ", "перспективные решения в финтехе",
"quantum sensing for navigation"...). Модуль:

  1. строит из темы несколько поисковых запросов, смещённых в сторону ранних
     стадий развития (раунды pre-seed/seed, пилоты, препринты, стартапы из
     stealth) — а не в сторону уже состоявшихся мейнстрим-технологий;
  2. рассылает эти запросы по всем доступным коннекторам (arXiv, Crossref,
     GDELT, Hacker News, GitHub, + опционально коммерческие API);
  3. для каждой находки — либо использует уже присланный API текст, либо
     догружает HTML-страницу через fetch.py и разбирает через parse.py;
  4. возвращает список SourceDocument, НЕ ограничиваясь стартовым XLSX и не
     ограничиваясь списком из 100 технологий: тема может быть любой.

Ошибка одного коннектора или одной страницы не прерывает весь сбор — она
логируется и попадает в отчёт `SearchRun.errors`, но остальной сбор
продолжается (см. ТЗ п. 2.2).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from typing import Optional

from .connectors import DEFAULT_CONNECTORS, OPTIONAL_CONNECTORS
from .connectors.base import ConnectorError, SearchConnector, SearchHit
from .fetch import Fetcher, get_default_fetcher
from .parse import parse_api_hit, parse_html_document
from .schemas import SourceDocument, SourceType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Построитель запросов из свободной темы
# --------------------------------------------------------------------------- #

# Модификаторы, смещающие поиск в сторону РАННИХ стадий развития технологии,
# а не уже состоявшегося мейнстрима. Используются как дополнительные
# варианты запроса, не заменяя базовый.
_EARLY_STAGE_MODIFIERS_RU = [
    "",  # базовый запрос без модификатора — тоже нужен
    "прототип",
    "пилот",
    "исследование",
    "стартап",
    "патент",
]

_EARLY_STAGE_MODIFIERS_EN = [
    "",
    "prototype",
    "pilot",
    "preprint",
    "startup",
    "patent",
]


def _looks_cyrillic(text: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


# Служебные слова, которые не считаются "содержательным" словом темы — иначе
# фильтр по перекрытию ключевых слов (см. _matches_topic) был бы бесполезен.
_STOPWORDS = {
    "без", "для", "или", "как", "что", "это", "при", "над", "под", "про",
    "все", "она", "оно", "они", "его", "тем", "том", "как", "так",
    "the", "for", "and", "or", "of", "in", "on", "to", "is", "are", "a", "an", "with",
}


def _topic_keywords(topic: str) -> list[str]:
    """Содержательные слова исходной темы (без учёта модификаторов стадии
    развития — они добавляются позже в build_queries и не должны считаться
    "темой" при проверке релевантности находки).
    """
    tokens = re.findall(r"\w+", topic.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _matches_topic(text: str, keywords: list[str]) -> bool:
    """Грубая лексическая проверка: находка должна содержать хотя бы одно
    содержательное слово исходной темы.

    Зачем это нужно (реальный, воспроизводимый дефект, а не гипотетический):
    поисковые API (arXiv, GDELT и т.п.) не понимают, что модификатор вида
    "стартап pre-seed seed" в запросе — это фильтр по стадии финансирования,
    а не самостоятельная тема поиска. В результате запрос
    "<тема> стартап pre-seed seed" у ряда коннекторов буквально матчится по
    слову "seed" и возвращает документы вообще не по теме (например, статьи
    про посевной материал в сельском хозяйстве или модель "Seed-Coder").
    Без этого фильтра такие документы попадали в кандидатов и получали
    неоправданно высокий rule-based score в extraction/relevance.py — то есть
    ошибка поиска маскировалась под "вероятный сигнал". Это НЕ фильтр
    "значимости" или "новизны" темы — чисто лексическая проверка совпадения
    со словами, которые сам пользователь ввёл в запрос.
    """
    if not keywords:
        return True
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def build_queries(topic: str, max_variants: int = 4) -> list[str]:
    """Строит несколько вариантов поискового запроса из свободной темы.

    Намеренно НЕ ограничивается никаким справочником технологий — тема может
    быть любой строкой на русском или английском (ТЗ п. 2.1: "Принимать
    произвольную технологическую тему").
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("Пустая тема запроса")

    modifiers = _EARLY_STAGE_MODIFIERS_RU if _looks_cyrillic(topic) else _EARLY_STAGE_MODIFIERS_EN

    queries = []
    for mod in modifiers[:max_variants]:
        q = f"{topic} {mod}".strip()
        if q not in queries:
            queries.append(q)
    return queries


# --------------------------------------------------------------------------- #
# Результат запуска поиска
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class ConnectorRunError:
    connector: str
    query: str
    message: str


@dataclasses.dataclass
class SearchRun:
    topic: str
    queries: list[str]
    documents: list[SourceDocument] = dataclasses.field(default_factory=list)
    errors: list[ConnectorRunError] = dataclasses.field(default_factory=list)
    connectors_used: list[str] = dataclasses.field(default_factory=list)
    connectors_skipped: list[str] = dataclasses.field(default_factory=list)
    filtered_low_relevance: int = 0  # находки без пересечения со словами темы (см. _matches_topic)
    budget_exhausted: bool = False
    queries_attempted: list[dict] = dataclasses.field(default_factory=list)

    def summary(self) -> dict:
        return {
            "topic": self.topic,
            "queries": self.queries,
            "documents_found": len(self.documents),
            "filtered_low_relevance": self.filtered_low_relevance,
            "connectors_used": self.connectors_used,
            "connectors_skipped": self.connectors_skipped,
            "errors": [dataclasses.asdict(e) for e in self.errors],
            "budget_exhausted": self.budget_exhausted,
            "queries_attempted": self.queries_attempted,
        }


# --------------------------------------------------------------------------- #
# Оркестратор
# --------------------------------------------------------------------------- #


class SearchOrchestrator:
    def __init__(
        self,
        connectors: Optional[list[SearchConnector]] = None,
        fetcher: Optional[Fetcher] = None,
        fetch_full_pages: bool = True,
        max_results_per_connector: int = 8,
    ):
        self.fetcher = fetcher or get_default_fetcher()
        if connectors is None:
            connectors = [cls() for cls in DEFAULT_CONNECTORS]
            connectors += [cls() for cls in OPTIONAL_CONNECTORS]
        self.connectors = connectors
        self.fetch_full_pages = fetch_full_pages
        self.max_results_per_connector = max_results_per_connector

    def run(self, topic: str, max_queries: int = 3, max_seconds: float | None = None) -> SearchRun:
        queries = build_queries(topic, max_variants=max_queries)
        run = SearchRun(topic=topic, queries=queries)
        topic_keywords = _topic_keywords(topic)

        seen_urls: set[str] = set()
        deadline = time.monotonic() + max_seconds if max_seconds is not None else None

        def exhausted():
            if deadline is not None and time.monotonic() >= deadline:
                run.budget_exhausted = True
                return True
            return False

        available = []
        for connector in self.connectors:
            if not connector.is_available():
                run.connectors_skipped.append(
                    f"{connector.name} (нет ключа/недоступен — {connector.description})"
                )
                continue
            available.append(connector)

        # Give every source the base query before spending time on variants.
        for query in queries:
            for connector in available:
                if exhausted():
                    return run
                if connector.name not in run.connectors_used:
                    run.connectors_used.append(connector.name)
                run.queries_attempted.append({'connector': connector.name, 'query': query})
                try:
                    hits = connector.search(query, max_results=self.max_results_per_connector)
                except ConnectorError as exc:
                    logger.warning("Коннектор %s не смог выполнить запрос %r: %s", connector.name, query, exc)
                    run.errors.append(ConnectorRunError(connector.name, query, str(exc)))
                    continue
                except Exception as exc:  # noqa: BLE001 — сбой одного коннектора не должен валить весь сбор
                    logger.exception("Неожиданная ошибка коннектора %s на запросе %r", connector.name, query)
                    run.errors.append(ConnectorRunError(connector.name, query, f"неожиданная ошибка: {exc}"))
                    continue

                for hit in hits:
                    if exhausted():
                        return run
                    for warning in hit.raw.get('discovery_warnings', []):
                        error = ConnectorRunError(connector.name, query, warning)
                        if error not in run.errors:
                            run.errors.append(error)
                    if not hit.url or hit.url in seen_urls:
                        continue
                    seen_urls.add(hit.url)

                    doc = self._to_document(hit, connector)
                    if doc is None:
                        run.errors.append(ConnectorRunError(connector.name, query, f'Could not parse {hit.url}'))
                        continue
                    if doc.fetch_status.value != 'ok':
                        run.errors.append(ConnectorRunError(connector.name, query, f'{hit.url}: {doc.fetch_status.value} {doc.fetch_error or ""}'))
                    if not _matches_topic(f"{doc.title or ''} {doc.text}", topic_keywords):
                        run.filtered_low_relevance += 1
                        continue
                    run.documents.append(doc)

        return run

    def _to_document(self, hit: SearchHit, connector: SearchConnector) -> Optional[SourceDocument]:
        default_type = SourceType(connector.default_source_type) if connector.default_source_type in SourceType._value2member_map_ else SourceType.UNKNOWN

        # Если API уже прислало полноценный текст (аннотация/описание) —
        # используем его напрямую, не делая лишний HTTP-запрос на сайт.
        if hit.full_text and len(hit.full_text) > 40:
            try:
                return parse_api_hit(hit, default_source_type=default_type)
            except Exception:  # noqa: BLE001
                logger.exception("Не удалось разобрать API-находку %s", hit.url)
                return None

        if not self.fetch_full_pages:
            try:
                return parse_api_hit(hit, default_source_type=default_type)
            except Exception:  # noqa: BLE001
                return None

        # Иначе — реальная загрузка страницы. Ошибка одной страницы (таймаут,
        # 404, блокировка) не прерывает сбор: parse_html_document в этом
        # случае вернёт SourceDocument с fetch_status != OK, который явно
        # виден дальше по конвейеру, а не молча теряется.
        try:
            fetch_result = self.fetcher.get(hit.url)
            doc = parse_html_document(fetch_result, hit)
            from .publication_metadata import recover_arxiv_date
            return recover_arxiv_date(doc, self.fetcher)
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось загрузить/разобрать страницу %s", hit.url)
            return None


def search_topic(topic: str, **kwargs) -> SearchRun:
    """Удобная функция верхнего уровня: `from retrieval.search import search_topic`."""
    orchestrator = SearchOrchestrator()
    return orchestrator.run(topic, **kwargs)
