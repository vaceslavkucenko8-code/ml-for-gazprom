"""
Общие схемы данных для конвейера Участника 2 (поиск, источники, извлечение).

Эти модели — контракт между:
  * поисковым/загрузочным слоем (search.py, fetch.py, parse.py, deduplicate.py)
  * слоем извлечения кандидатов (extraction/candidates.py)
  * Участником 1 (модель классификации слабых сигналов), который получает
    объекты Candidate и НЕ получает от нас финальную метку "это слабый сигнал".
  * Участником 3 (аналитика/интерфейс), который получает SourceDocument
    в согласованной схеме.

Все модели — pydantic v2 BaseModel, поэтому:
  * есть валидация типов "из коробки";
  * есть .model_dump() / .model_dump_json() для сериализации в API/БД;
  * схему можно напрямую использовать как response_model в FastAPI (см. ТЗ,
    стек backend — FastAPI).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_doc_id(normalized_url: str) -> str:
    """Детерминированный id документа: один и тот же URL всегда даёт один и тот
    же doc_id. Это важно для дедупликации между запусками (см. deduplicate.py).
    """
    h = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    return f"doc_{h}"


# --------------------------------------------------------------------------- #
# Перечисления
# --------------------------------------------------------------------------- #


class SourceType(str, Enum):
    """Тип источника. Соответствует классификации из ТЗ (п. "Требования к
    источникам"): первоисточник / публикация разработчика / научная работа /
    отраслевое медиа / пресс-релиз / агрегатор — плюс несколько подтипов,
    которые нужны для корректной оценки доверия.
    """

    PRIMARY_OFFICIAL = "primary_official"  # гос. орган, регулятор, межд. организация, реестр
    PATENT = "patent"  # патентная база
    ACADEMIC = "academic_paper"  # научная публикация, препринт, материалы конференции
    UNIVERSITY = "university"  # institutional research news, not a peer-reviewed paper
    DEVELOPER_PUBLICATION = "developer_publication"  # официальный сайт/блог компании-разработчика
    ANALYST_REPORT = "analyst_report"  # аналитический отчёт (Gartner-подобные, институты)
    INDUSTRY_MEDIA = "industry_media"  # профессиональное отраслевое медиа
    GENERAL_MEDIA = "general_media"  # общее новостное медиа
    PRESS_RELEASE = "press_release"  # пресс-релиз компании/PR-агентства
    AGGREGATOR = "aggregator"  # агрегатор новостей, синдикация без добавленной ценности
    SOCIAL_MEDIA = "social_media"
    PERSONAL_BLOG = "personal_blog"
    FORUM = "forum"
    CODE_REPOSITORY = "code_repository"  # GitHub/GitLab и т.п. — сигнал разработческой активности
    UNKNOWN = "unknown"


class TrustLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


class FetchStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    CONNECTION_ERROR = "connection_error"
    PARSE_ERROR = "parse_error"
    BLOCKED = "blocked"  # robots.txt / 403 / paywall
    SKIPPED = "skipped"


class ClaimType(str, Enum):
    STAGE_OF_DEVELOPMENT = "stage_of_development"
    ADVANTAGE = "advantage"
    CASE_EXAMPLE = "case_example"
    COMPANY_MENTION = "company_mention"
    GENERAL_FACT = "general_fact"


# --------------------------------------------------------------------------- #
# SourceDocument
# --------------------------------------------------------------------------- #


class SourceDocument(BaseModel):
    """Единица результата загрузки: один документ (статья, препринт, запись
    патента, пресс-релиз и т.д.), приведённый к общей схеме.

    Поле `published_at=None` — это ЯВНОЕ обозначение "дата публикации
    неизвестна". Мы никогда не подставляем вместо неё `fetched_at` — это
    отдельное требование ТЗ (п. 2.2): "не подменять его временем загрузки".
    """

    doc_id: str
    url: str
    normalized_url: str

    title: Optional[str] = None
    original_title: Optional[str] = None  # заголовок на языке оригинала, если title — перевод

    text: str = ""
    text_excerpt: Optional[str] = None  # короткий фрагмент для превью (не для доказательства)

    language: Optional[str] = None  # ISO 639-1, напр. "ru", "en"
    language_confidence: Optional[float] = None

    source_type: SourceType = SourceType.UNKNOWN
    source_type_confidence: Optional[float] = None
    source_type_rationale: Optional[str] = None
    source_name: Optional[str] = None  # человекочитаемое имя издания/организации
    domain: Optional[str] = None  # доменное имя (для группировки/policy)

    published_at: Optional[datetime] = None
    published_at_is_estimated: bool = False
    fetched_at: datetime = Field(default_factory=now_utc)

    connector: str = "unknown"  # какой коннектор/источник нашёл документ
    query: Optional[str] = None  # исходный поисковый запрос (свободная тема)

    trust_level: Optional[TrustLevel] = None
    trust_explanation: Optional[str] = None

    is_translated: bool = False
    translated_from: Optional[str] = None
    is_generated_summary: bool = False

    fetch_status: FetchStatus = FetchStatus.OK
    fetch_error: Optional[str] = None

    content_hash: Optional[str] = None  # для точного/почти точного дедупа
    duplicate_of: Optional[str] = None  # doc_id канонического документа, если это дубль
    event_cluster_id: Optional[str] = None  # см. deduplicate.py: кластер "один и тот же инфоповод"

    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text_excerpt", mode="after")
    @classmethod
    def _fill_excerpt(cls, v, info):
        if v:
            return v
        text = info.data.get("text") or ""
        return (text[:400] + "…") if len(text) > 400 else (text or None)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    """Атомарная единица доказательства: конкретная цитата из конкретного
    документа, подтверждающая конкретное утверждение о кандидате.

    ТЗ (п. 2.4): "для каждого факта сохранять ссылку на документ и
    подтверждающий фрагмент" — это ровно то, что делает Evidence.quote +
    Evidence.doc_id/url.
    """

    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    doc_id: str
    url: str
    quote: str  # дословная цитата из текста документа
    context: Optional[str] = None  # 1-2 предложения вокруг цитаты, для контекста

    claim_type: ClaimType = ClaimType.GENERAL_FACT
    is_future_looking: bool = False  # True = план/анонс на будущее, False = свершившийся факт

    published_at: Optional[datetime] = None
    source_type: SourceType = SourceType.UNKNOWN
    trust_level: Optional[TrustLevel] = None

    is_translated: bool = False
    is_generated_summary: bool = False


class DevelopmentStageFact(BaseModel):
    """Факт о стадии развития технологии (раунд финансирования, пилот, патент,
    публикация, регуляторное одобрение и т.п.)."""

    label: str  # напр. "pre-seed финансирование", "пилотное внедрение", "патентная заявка"
    description: str
    is_future_looking: bool = False
    evidence_ids: list[str] = Field(default_factory=list)


class Advantage(BaseModel):
    """Подтверждённое преимущество технологии (п. 2.5)."""

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class CaseExample(BaseModel):
    """Кейс-пример применения (п. 2.5)."""

    text: str
    organization: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Candidate
# --------------------------------------------------------------------------- #


class Candidate(BaseModel):
    """Кандидат в слабый сигнал — то, что Участник 2 передаёт Участнику 1.

    ВАЖНО: здесь намеренно нет поля вроде `is_weak_signal` или `final_score`.
    Участник 2 не присваивает финальный класс — это задача модели Участника 1
    (см. ТЗ и формулировку задачи 2.4: "не назначая самостоятельно финальный
    класс модели"). Мы поставляем только проверяемые факты и их источники.
    """

    candidate_id: str = Field(default_factory=lambda: new_id("cand"))
    technology_name: str
    short_description: str
    domain: Optional[str] = None  # область, напр. "Защита ИИ", "Edge", "Финтех"
    companies: list[str] = Field(default_factory=list)

    query: str  # свободный запрос пользователя, в рамках которого найден кандидат

    stage_facts: list[DevelopmentStageFact] = Field(default_factory=list)
    advantages: list[Advantage] = Field(default_factory=list)
    case_examples: list[CaseExample] = Field(default_factory=list)

    evidence: list[Evidence] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)

    independent_confirmations: int = 0
    confirmation_explanation: Optional[str] = None

    lowest_trust_only: bool = False  # True = держится только на LOW-доверенных источниках
    trust_summary: Optional[str] = None

    created_at: datetime = Field(default_factory=now_utc)
    notes: Optional[str] = None
