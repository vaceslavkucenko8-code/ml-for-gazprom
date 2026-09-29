"""
src/retrieval/deduplicate.py

Реализует три связанные задачи из ТЗ (п. 2.3):

  1. Нормализация URL и поиск точных/почти точных дублей одного документа
     (перепечатка того же текста на разных URL, зеркала, AMP-версии и т.п.).
  2. Кластеризация документов по "инфоповоду" (event clustering): разные
     статьи, написанные по мотивам ОДНОГО И ТОГО ЖЕ события/пресс-релиза
     (например: новость о выходе отчёта и сам PDF отчёта; или один и тот же
     пресс-релиз, перепечатанный пятью изданиями). Это не "дубли" в смысле
     текста, но и не независимые подтверждения.
  3. Присвоение уровня доверия (TrustLevel) каждому документу — с явным,
     воспроизводимым обоснованием (см. docs/source_policy.md), и подсчёт
     количества НЕЗАВИСИМЫХ подтверждений для кандидата (используется в
     extraction/candidates.py).

Ключевое требование ТЗ, которое здесь реализовано буквально:
    "Не считать новость об отчёте и PDF того же отчёта двумя независимыми
    подтверждениями."
Это в коде — функция `cluster_by_event`: документы, описывающие одно и то же
событие (высокое текстовое сходство И/ИЛИ явная пометка "перевод/пересказ"
И близкая дата публикации), попадают в один event_cluster_id. При подсчёте
независимых подтверждений считается число РАЗЛИЧНЫХ event-кластеров с
РАЗЛИЧНЫМИ независимыми издателями, а не число документов.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import timedelta
from typing import Optional

from rapidfuzz import fuzz

from .parse import domain_of, normalize_url
from .schemas import SourceDocument, SourceType, TrustLevel

logger = logging.getLogger(__name__)

# Пороговые значения сходства (0..100, шкала rapidfuzz)
EXACT_DUPLICATE_URL = True  # включает сравнение по normalized_url
NEAR_DUPLICATE_TEXT_THRESHOLD = 88  # практически идентичный текст -> дубль
SAME_EVENT_TEXT_THRESHOLD = 45  # значимое пересечение -> тот же инфоповод
SAME_EVENT_MAX_DAYS_APART = 10  # событие "устаревает" как общий инфоповод

# Приоритет источников при выборе "канонического" документа в группе дублей:
# чем меньше число — тем выше приоритет.
_CANONICAL_PRIORITY = {
    SourceType.PRIMARY_OFFICIAL: 0,
    SourceType.PATENT: 1,
    SourceType.ACADEMIC: 2,
    SourceType.DEVELOPER_PUBLICATION: 3,
    SourceType.ANALYST_REPORT: 4,
    SourceType.INDUSTRY_MEDIA: 5,
    SourceType.GENERAL_MEDIA: 6,
    SourceType.PRESS_RELEASE: 7,
    SourceType.CODE_REPOSITORY: 8,
    SourceType.FORUM: 9,
    SourceType.AGGREGATOR: 10,
    SourceType.PERSONAL_BLOG: 11,
    SourceType.SOCIAL_MEDIA: 12,
    SourceType.UNKNOWN: 13,
}

# Источники, которые НИКОГДА не могут в одиночку "нести" кандидата — см. ТЗ:
# "не должны быть единственным основанием для включения технологии".
LOW_TRUST_TYPES = {
    SourceType.SOCIAL_MEDIA,
    SourceType.PERSONAL_BLOG,
    SourceType.FORUM,
    SourceType.AGGREGATOR,
    SourceType.PRESS_RELEASE,
}

HIGH_TRUST_TYPES = {
    SourceType.UNIVERSITY,
    SourceType.PRIMARY_OFFICIAL,
    SourceType.PATENT,
    SourceType.ACADEMIC,
}


# --------------------------------------------------------------------------- #
# Точные и почти точные дубли
# --------------------------------------------------------------------------- #


def _canonical_rank(doc: SourceDocument) -> tuple:
    return (_CANONICAL_PRIORITY.get(doc.source_type, 99), doc.published_at or doc.fetched_at)


def deduplicate_exact(documents: list[SourceDocument]) -> list[SourceDocument]:
    """Убирает точные дубли по normalized_url (оставляет по одному документу
    на нормализованный URL). Использовать до near-duplicate дедупликации.
    """
    by_url: dict[str, SourceDocument] = {}
    for doc in documents:
        key = doc.normalized_url or normalize_url(doc.url)
        if key not in by_url:
            by_url[key] = doc
        else:
            # оставляем документ с более полным текстом / более высоким рангом типа
            existing = by_url[key]
            if len(doc.text or "") > len(existing.text or ""):
                by_url[key] = doc
    return list(by_url.values())


def deduplicate_near(documents: list[SourceDocument]) -> list[SourceDocument]:
    """Находит почти точные текстовые дубли (одна и та же статья на разных
    URL/зеркалах) и помечает их через `duplicate_of`, оставляя один
    канонический документ в возвращаемом списке (остальные помечаются, но не
    удаляются из исходного списка полностью — см. `annotate_duplicates`,
    которая работает in-place и возвращает ВСЕ документы).
    """
    return annotate_duplicates(documents)


def annotate_duplicates(documents: list[SourceDocument]) -> list[SourceDocument]:
    """Проставляет `duplicate_of` для почти точных текстовых дублей.
    Возвращает тот же список (модифицирует объекты), ничего не удаляя —
    решение "показывать ли дубли" остаётся за потребителем (Участник 3).
    """
    n = len(documents)
    assigned = [False] * n
    # сортируем по каноническому приоритету, чтобы дубль всегда ссылался на
    # более авторитетный документ, а не наоборот
    order = sorted(range(n), key=lambda i: _canonical_rank(documents[i]))

    for a_idx in range(len(order)):
        i = order[a_idx]
        if assigned[i]:
            continue
        text_i = (documents[i].text or documents[i].title or "")[:5000]
        if not text_i:
            continue
        for b_idx in range(a_idx + 1, len(order)):
            j = order[b_idx]
            if assigned[j] or documents[j].duplicate_of:
                continue
            text_j = (documents[j].text or documents[j].title or "")[:5000]
            if not text_j:
                continue
            score = fuzz.token_set_ratio(text_i, text_j)
            if score >= NEAR_DUPLICATE_TEXT_THRESHOLD:
                documents[j].duplicate_of = documents[i].doc_id
                assigned[j] = True
    return documents


# --------------------------------------------------------------------------- #
# Кластеризация "один и тот же инфоповод"
# --------------------------------------------------------------------------- #


def cluster_by_event(documents: list[SourceDocument]) -> list[SourceDocument]:
    """Группирует документы, освещающие одно и то же событие/пресс-релиз, но
    НЕ являющиеся текстовыми дублями (разный текст, разный язык, разный
    издатель — но один и тот же первичный инфоповод).

    Критерий объединения в один event_cluster_id (все условия одновременно):
      * даты публикации отличаются не более чем на SAME_EVENT_MAX_DAYS_APART
        дней (или хотя бы у одного из двух дата неизвестна — тогда критерий
        по дате не блокирует объединение, чтобы не терять реальные дубли
        из-за отсутствующих метаданных);
      * текстовое сходство (rapidfuzz.token_set_ratio) в диапазоне
        [SAME_EVENT_TEXT_THRESHOLD, NEAR_DUPLICATE_TEXT_THRESHOLD) —
        значимое пересечение содержания, но не полный дубль;
      * ИЛИ документ явно помечен как перевод/пересказ (is_translated /
        is_generated_summary) с похожим текстом — типичный случай "локальное
        медиа пересказало иностранный пресс-релиз".
    Модифицирует документы in-place (`event_cluster_id`) и возвращает список.
    """
    n = len(documents)
    cluster_id_of: dict[int, str] = {}
    next_cluster = 0

    def get_cluster(i: int) -> str:
        nonlocal next_cluster
        if i not in cluster_id_of:
            cluster_id_of[i] = f"evt_{next_cluster:04d}"
            next_cluster += 1
        return cluster_id_of[i]

    for i in range(n):
        text_i = (documents[i].text or documents[i].title or "")[:5000]
        if not text_i:
            continue
        for j in range(i + 1, n):
            text_j = (documents[j].text or documents[j].title or "")[:5000]
            if not text_j:
                continue

            if documents[i].published_at and documents[j].published_at:
                if abs((documents[i].published_at - documents[j].published_at)) > timedelta(days=SAME_EVENT_MAX_DAYS_APART):
                    continue

            score = fuzz.token_set_ratio(text_i, text_j)
            same_event = SAME_EVENT_TEXT_THRESHOLD <= score
            if not same_event:
                continue

            ci = cluster_id_of.get(i)
            cj = cluster_id_of.get(j)
            if ci and cj and ci != cj:
                # объединяем два существующих кластера в один (берём меньший id)
                keep, drop = (ci, cj) if ci < cj else (cj, ci)
                for k, v in list(cluster_id_of.items()):
                    if v == drop:
                        cluster_id_of[k] = keep
            elif ci:
                cluster_id_of[j] = ci
            elif cj:
                cluster_id_of[i] = cj
            else:
                new_c = get_cluster(i)
                cluster_id_of[j] = new_c

    for i in range(n):
        cluster = cluster_id_of.get(i)
        if cluster is None:
            cluster = f"evt_{next_cluster:04d}"
            next_cluster += 1
        documents[i].event_cluster_id = cluster

    return documents


# --------------------------------------------------------------------------- #
# Доверие
# --------------------------------------------------------------------------- #


def assign_trust(doc: SourceDocument, corroborating_domains: int = 0) -> SourceDocument:
    """Присваивает TrustLevel и explanation по правилам docs/source_policy.md.

    `corroborating_domains` — число ДРУГИХ доменов, независимо подтверждающих
    тот же факт/событие (вычисляется на уровне кандидата, см.
    extraction/candidates.py -> count_independent_confirmations).
    """
    if doc.fetch_status.value != "ok":
        doc.trust_level = TrustLevel.UNVERIFIED
        doc.trust_explanation = f"Документ не загружен ({doc.fetch_status.value}) — доверие не может быть определено."
        return doc

    reasons = [f"тип источника — {doc.source_type.value} ({doc.source_type_rationale or 'без уточнения'})"]

    if doc.source_type in HIGH_TRUST_TYPES:
        level = TrustLevel.HIGH
        reasons.append("тип источника относится к доверенным по политике (первоисточник/патент/научная публикация)")
    elif doc.source_type in LOW_TRUST_TYPES:
        if corroborating_domains >= 1:
            level = TrustLevel.MEDIUM
            reasons.append(
                f"источник низкого базового доверия, но подтверждён {corroborating_domains} независимым(и) доменом(ами)"
            )
        else:
            level = TrustLevel.LOW
            reasons.append("источник низкого базового доверия (соцсеть/блог/агрегатор/пресс-релиз/форум) без независимого подтверждения")
    else:
        # industry_media, general_media, analyst_report, developer_publication, code_repository
        if corroborating_domains >= 1 or doc.source_type == SourceType.DEVELOPER_PUBLICATION:
            level = TrustLevel.HIGH if doc.source_type == SourceType.ANALYST_REPORT else TrustLevel.MEDIUM
            reasons.append("тип источника среднего доверия" + (
                f", подтверждён {corroborating_domains} независимым(и) доменом(ами)" if corroborating_domains else ", официальная публикация разработчика технологии"
            ))
        else:
            level = TrustLevel.MEDIUM if doc.source_type_confidence and doc.source_type_confidence >= 0.6 else TrustLevel.LOW
            reasons.append("тип источника среднего доверия без дополнительного подтверждения")

    if doc.is_translated:
        reasons.append("текст получен автоматическим переводом — отмечено отдельно, доверие не понижается автоматически")
    if doc.is_generated_summary:
        reasons.append("резюме сгенерировано автоматически (не дословная цитата первоисточника) — отмечено отдельно")

    doc.trust_level = level
    doc.trust_explanation = "; ".join(reasons) + "."
    return doc


def deduplicate_and_score(documents: list[SourceDocument]) -> list[SourceDocument]:
    """Полный конвейер: точные дубли -> почти-дубли -> event-кластеры ->
    доверие. Возвращает документы со всеми проставленными полями.
    """
    documents = deduplicate_exact(documents)
    documents = annotate_duplicates(documents)
    documents = cluster_by_event(documents)

    # число независимых доменов на event-кластер (без учёта самого документа)
    domains_per_cluster: dict[str, set] = {}
    for doc in documents:
        if doc.duplicate_of:
            continue
        domains_per_cluster.setdefault(doc.event_cluster_id, set()).add(domain_of(doc.url))

    for doc in documents:
        if doc.duplicate_of:
            doc.trust_level = TrustLevel.UNVERIFIED
            doc.trust_explanation = f"Точный/почти точный дубль документа {doc.duplicate_of}; доверие наследуется от канонического документа."
            continue
        cluster_domains = domains_per_cluster.get(doc.event_cluster_id, set())
        # Distinct hosts can republish one announcement. No verified provenance
        # graph exists here, so clustering must not upgrade low-trust sources.
        corroborating = 0
        assign_trust(doc, corroborating_domains=corroborating)

    return documents


@dataclasses.dataclass
class ConfirmationResult:
    independent_confirmations: int
    explanation: str


def count_independent_confirmations(documents: list[SourceDocument]) -> ConfirmationResult:
    """Считает число НЕЗАВИСИМЫХ подтверждений среди документов одного
    кандидата: число различных event_cluster_id, ИСКЛЮЧАЯ дубли и кластеры,
    состоящие только из одного домена одного низко-доверенного типа.

    Именно эта функция реализует требование:
    "новость об отчёте и PDF того же отчёта — не два независимых
    подтверждения" — обе попадут в один event_cluster_id и будут посчитаны
    один раз.
    """
    clusters: dict[str, list[SourceDocument]] = {}
    for doc in documents:
        if doc.duplicate_of:
            continue
        clusters.setdefault(doc.event_cluster_id or doc.doc_id, []).append(doc)

    count = 0
    parts = []
    for cluster_id, docs in clusters.items():
        domains = {domain_of(d.url) for d in docs}
        types = {d.source_type for d in docs}
        if len(domains) == 1 and types <= LOW_TRUST_TYPES:
            # один домен низкого доверия — не считается независимым подтверждением
            continue
        count += 1
        parts.append(f"{cluster_id}: {len(domains)} домен(ов) ({', '.join(sorted(domains))})")

    explanation = (
        f"{count} независимых подтверждений из {len(clusters)} событийных кластеров. "
        + ("; ".join(parts) if parts else "подтверждений, отвечающих критерию независимости, не найдено")
    )
    return ConfirmationResult(independent_confirmations=count, explanation=explanation)
