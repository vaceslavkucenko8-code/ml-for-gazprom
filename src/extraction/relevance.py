"""
src/extraction/relevance.py

Автоматическая, объяснимая оценка кандидата: "похоже на зарождающийся
технологический сигнал" / "похоже на шум" / "пока не хватает данных" — и
отдельно "актуален ли он сейчас или это устаревшее упоминание".

ВАЖНО (сохраняет принцип из docs/design_principles.md): это НЕ обращение к
готовой ML-библиотеке "детекции трендов" и не предобученный классификатор.
Это явный, документированный набор правил ("если признак X — то вклад в
итоговый счёт Y, потому что Z"), построенный НАД уже посчитанными прозрачными
признаками из `extraction/features.py`. Каждое правило можно прочитать,
объяснить жюри и поправить — в отличие от "чёрного ящика".

Зачем этот модуль нужен (в дополнение к features.py):
  * `features.py` только СЧИТАЕТ признаки и не принимает решения;
  * пользователю модуля поиска ("Участнику 2" в контексте хакатона, либо
    любому, кто просто запускает поиск по теме) нужен готовый, а не
    сырой ответ: не бегать вручную по кандидатам, а сразу увидеть вердикт
    с объяснением. Финальная модель Участника 1 (если она есть в проекте)
    остаётся более точным и обучаемым слоем НАД этим; этот модуль — прозрачный
    поведенческий default, который работает даже без обученной модели.

"Нормализация" в данном модуле означает: разнородные, разнонаправленные
признаки (целые числа подтверждений, доли [0..1], давность в днях) сводятся
к ЕДИНОЙ сопоставимой шкале score in [0, 1] по явной, документированной
формуле — а не "нормализация текста/URL" (это уже сделано в parse.py/
deduplicate.py на уровне документов).

"Актуальность" (`is_actual`) оценивается ОТДЕЛЬНО от общего score: по тому,
насколько свежа самая новая датированная публикация о кандидате относительно
текущего момента. Кандидат может быть "похож на настоящий сигнал" (высокий
score), но при этом "неактуален сейчас" (последнее упоминание — полтора года
назад и не подтверждается ничем новым) — это два разных вопроса, поэтому они
не смешиваются в одно число.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..retrieval.schemas import Candidate, SourceDocument, now_utc
from .features import candidate_features

# --------------------------------------------------------------------------- #
# Пороги и веса — вынесены в константы, чтобы решение было настраиваемым и
# объяснимым числом, а не "магией" внутри функции.
# --------------------------------------------------------------------------- #

# Свежесть последнего датированного упоминания относительно текущего момента.
FRESH_WINDOW_DAYS = 180  # <= 6 месяцев — тема явно на слуху сейчас
STALE_WINDOW_DAYS = 540  # > ~1.5 года без новых подтверждений — вероятно, неактуально

# Веса правил скоринга (сумма положительных вкладов ограничена, чтобы одно
# правило не могло единолично "перевесить" всё остальное).
WEIGHT_NO_CONFIRMATION = -0.35
WEIGHT_ONE_CONFIRMATION = 0.10
WEIGHT_MULTI_CONFIRMATION = 0.30
WEIGHT_LOWEST_TRUST_ONLY = -0.25
WEIGHT_ACADEMIC_OR_PATENT = 0.15
WEIGHT_HIGH_TRUST_SOURCE = 0.10
WEIGHT_EARLY_STAGE_MAJORITY = 0.10
WEIGHT_ALL_FUTURE_LOOKING = -0.05
WEIGHT_HAS_REALIZED_FACT = 0.05
WEIGHT_HAS_ADVANTAGE_OR_CASE = 0.05
WEIGHT_FRESH = 0.15
WEIGHT_STALE = -0.20

SCORE_BASELINE = 0.5  # нейтральная стартовая точка перед применением правил

VERDICT_SIGNAL_THRESHOLD = 0.65
VERDICT_NOISE_THRESHOLD = 0.40


class RelevanceVerdict(str, Enum):
    LIKELY_EMERGING_SIGNAL = "likely_emerging_signal"
    WORTH_MONITORING = "worth_monitoring"
    LIKELY_NOISE = "likely_noise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class RuleContribution(BaseModel):
    """Один явный вклад в итоговый счёт — для объяснения "почему именно так"."""

    rule: str
    weight: float
    reason: str


class RelevanceAssessment(BaseModel):
    """Результат прозрачной, rule-based оценки одного кандидата.

    Это НЕ замена `Candidate` и не модифицирует его — намеренно отдельный
    объект (см. schemas.py: у `Candidate` нет и не должно быть полей вроде
    `is_weak_signal`/`final_score`). `RelevanceAssessment` — необязательный,
    прикладной слой поверх кандидата, который можно как использовать сразу,
    так и полностью заменить обученной моделью без изменения остального кода.
    """

    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    verdict: RelevanceVerdict
    is_actual: Optional[bool]  # True/False/None (недостаточно данных о датах)
    actuality_reason: str
    contributions: list[RuleContribution]
    computed_at: datetime = Field(default_factory=now_utc)


def _latest_published_at(candidate: Candidate, documents: Optional[list[SourceDocument]]) -> Optional[datetime]:
    docs_by_id = {d.doc_id: d for d in (documents or [])}
    relevant = [docs_by_id[i] for i in candidate.source_doc_ids if i in docs_by_id]
    dates = [d.published_at for d in relevant if d.published_at and not d.duplicate_of]
    if not dates:
        dates = [e.published_at for e in candidate.evidence if e.published_at]
    return max(dates) if dates else None


def assess_relevance(
    candidate: Candidate,
    documents: Optional[list[SourceDocument]] = None,
    *,
    now: Optional[datetime] = None,
) -> RelevanceAssessment:
    """Считает прозрачный score/вердикт/актуальность для одного кандидата.

    Ничего не скрыто: каждое правило добавлено в `contributions` с весом и
    текстовым обоснованием — этот список можно напрямую показать в UI как
    "почему алгоритм так решил" (см. `explain_relevance`).
    """
    now = now or now_utc()
    features = candidate_features(candidate, documents)
    contributions: list[RuleContribution] = []
    score = SCORE_BASELINE

    confirmations = features["independent_confirmations"]
    if confirmations == 0:
        w, reason = WEIGHT_NO_CONFIRMATION, (
            "нет ни одного независимого подтверждения (все документы — один "
            "инфоповод/домен) — типичный профиль шума или маркетингового вброса"
        )
    elif confirmations == 1:
        w, reason = WEIGHT_ONE_CONFIRMATION, (
            "есть ровно 1 независимое подтверждение — сигнал существует, но "
            "пока слабо подтверждён"
        )
    else:
        w, reason = WEIGHT_MULTI_CONFIRMATION, (
            f"{confirmations} независимых подтверждения из разных инфоповодов/доменов "
            "— несколько независимых источников со временем типично для "
            "формирующегося тренда, а не для шума"
        )
    contributions.append(RuleContribution(rule="independent_confirmations", weight=w, reason=reason))
    score += w

    if features["lowest_trust_only"]:
        w = WEIGHT_LOWEST_TRUST_ONLY
        contributions.append(RuleContribution(
            rule="lowest_trust_only",
            weight=w,
            reason=(
                "держится исключительно на источниках низкого доверия "
                "(соцсети/блоги/агрегаторы/пресс-релизы/форумы) без независимого "
                "подтверждения — по правилам источников это не может быть "
                "единственным основанием"
            ),
        ))
        score += w

    if features["has_academic_or_patent_evidence"]:
        w = WEIGHT_ACADEMIC_OR_PATENT
        contributions.append(RuleContribution(
            rule="has_academic_or_patent_evidence",
            weight=w,
            reason="подтверждено научной публикацией или патентной заявкой — проверяемый первоисточник",
        ))
        score += w

    if features["high_trust_source_count"] and features["high_trust_source_count"] > 0:
        w = WEIGHT_HIGH_TRUST_SOURCE
        contributions.append(RuleContribution(
            rule="high_trust_source_count",
            weight=w,
            reason=f"{features['high_trust_source_count']} источник(ов) с доверием HIGH",
        ))
        score += w

    early_ratio = features["early_stage_marker_ratio"]
    if early_ratio is not None and early_ratio >= 0.5:
        w = WEIGHT_EARLY_STAGE_MAJORITY
        contributions.append(RuleContribution(
            rule="early_stage_marker_ratio",
            weight=w,
            reason=(
                f"{early_ratio:.0%} фактов о стадии развития относятся к ранним "
                "стадиям (pre-seed/pilot/patent) — характерно для раннего сигнала, "
                "а не для уже сформированного рынка"
            ),
        ))
        score += w

    future_ratio = features["future_looking_ratio"]
    if future_ratio is not None:
        if future_ratio >= 0.999:
            w = WEIGHT_ALL_FUTURE_LOOKING
            contributions.append(RuleContribution(
                rule="future_looking_ratio",
                weight=w,
                reason="абсолютно все свидетельства — планы/анонсы на будущее, ни одного свершившегося факта",
            ))
            score += w
        else:
            w = WEIGHT_HAS_REALIZED_FACT
            contributions.append(RuleContribution(
                rule="future_looking_ratio",
                weight=w,
                reason="есть хотя бы один свершившийся факт, а не только анонс планов",
            ))
            score += w

    if features["has_advantage_evidence"] or features["has_case_example_evidence"]:
        w = WEIGHT_HAS_ADVANTAGE_OR_CASE
        contributions.append(RuleContribution(
            rule="has_advantage_evidence_or_case_example",
            weight=w,
            reason="есть подтверждённое текстом преимущество и/или кейс-пример применения, а не только упоминание названия",
        ))
        score += w

    latest_date = _latest_published_at(candidate, documents)
    if latest_date is None:
        is_actual = None
        actuality_reason = (
            "дата публикации не установлена ни у одного документа кандидата — "
            "оценить актуальность во времени невозможно, только сам факт наличия сигнала"
        )
    else:
        # некоторые источники/фикстуры отдают naive datetime — считаем его UTC,
        # чтобы вычитание из tz-aware `now` не падало (см. README про
        # аналогичный фикс naive/aware в deduplicate.py)
        latest_date_aware = latest_date if latest_date.tzinfo else latest_date.replace(tzinfo=timezone.utc)
        age_days = (now - latest_date_aware).days
        if age_days <= FRESH_WINDOW_DAYS:
            is_actual = True
            w = WEIGHT_FRESH
            actuality_reason = (
                f"самая свежая публикация о кандидате — {age_days} дн. назад "
                f"(≤ {FRESH_WINDOW_DAYS} дн.) — тема на слуху сейчас"
            )
            contributions.append(RuleContribution(rule="freshness", weight=w, reason=actuality_reason))
            score += w
        elif age_days <= STALE_WINDOW_DAYS:
            is_actual = None
            actuality_reason = (
                f"самая свежая публикация о кандидате — {age_days} дн. назад "
                f"(между {FRESH_WINDOW_DAYS} и {STALE_WINDOW_DAYS} дн.) — ни явно "
                "свежо, ни явно устарело, нужно доследить"
            )
        else:
            is_actual = False
            w = WEIGHT_STALE
            actuality_reason = (
                f"самая свежая публикация о кандидате — {age_days} дн. назад "
                f"(> {STALE_WINDOW_DAYS} дн.) без новых подтверждений — вероятно, "
                "интерес к теме угас или сигнал устарел"
            )
            contributions.append(RuleContribution(rule="freshness", weight=w, reason=actuality_reason))
            score += w

    score = max(0.0, min(1.0, score))

    if confirmations == 0 and latest_date is None:
        verdict = RelevanceVerdict.INSUFFICIENT_EVIDENCE
    elif score >= VERDICT_SIGNAL_THRESHOLD:
        verdict = RelevanceVerdict.LIKELY_EMERGING_SIGNAL
    elif score >= VERDICT_NOISE_THRESHOLD:
        verdict = RelevanceVerdict.WORTH_MONITORING
    else:
        verdict = RelevanceVerdict.LIKELY_NOISE

    return RelevanceAssessment(
        candidate_id=candidate.candidate_id,
        score=score,
        verdict=verdict,
        is_actual=is_actual,
        actuality_reason=actuality_reason,
        contributions=contributions,
    )


def assess_candidates(
    candidates: list[Candidate],
    documents: Optional[list[SourceDocument]] = None,
    *,
    now: Optional[datetime] = None,
) -> list[RelevanceAssessment]:
    """Пакетная версия `assess_relevance` — то, что реально дёргает "боевой"
    запуск поиска, чтобы не заставлять пользователя вручную разбирать
    каждого кандидата.
    """
    return [assess_relevance(c, documents, now=now) for c in candidates]


def explain_relevance(assessment: RelevanceAssessment) -> list[str]:
    """Строки вида 'правило (вес): причина' — для прямого показа в UI/CLI,
    без обращения к автору кода."""
    lines = [
        f"score = {assessment.score:.2f}; вердикт = {assessment.verdict.value}; "
        f"актуально сейчас = {assessment.is_actual} ({assessment.actuality_reason})"
    ]
    for c in assessment.contributions:
        sign = "+" if c.weight >= 0 else ""
        lines.append(f"  {c.rule} ({sign}{c.weight:.2f}): {c.reason}")
    return lines
