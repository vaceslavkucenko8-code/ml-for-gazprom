"""
src/extraction/features.py

ВАЖНО (инженерная ценность решения): этот модуль НЕ классифицирует кандидата
как "слабый сигнал". Он превращает уже собранные, проверяемые факты
(`Candidate`, `SourceDocument`) в набор ПРОЗРАЧНЫХ, поштучно объяснимых
числовых/булевых признаков — своего рода "витрину предикторов", которую
Участник 1 использует как вход для интерпретируемой модели (например,
логистическая регрессия / дерево решений с понятными весами), не прибегая ни
к какой готовой библиотеке "детекции трендов" или предобученному классификатору.

Почему это важно и сделано именно так: ценность решения — это то, как
программа отличает "редкое упоминание в научной статье = зарождающийся
тренд" от "редкое упоминание = шум". Если бы этот выбор скрывался внутри
чьей-то сторонней библиотеки (готовый "trend detector", topic modeling
из коробки,預train классификатор новизны и т.п.), команда теряла бы именно
эту инженерную ценность — умение объяснить и обосновать критерий отбора
признаков (см. ТЗ, п. 8.3: "Интерпретируемость модели: прозрачность
признаков, по которым алгоритм относит наблюдение к сигналу").

Поэтому каждый признак здесь:
  * вычисляется прямым, читаемым кодом на собственных структурах данных
    (Candidate/SourceDocument), без сторонних ML/NLP библиотек;
  * сопровождается описанием (см. FEATURE_DESCRIPTIONS) — почему он вообще
    может отличать зарождающийся тренд от шума;
  * не принимает решения сам — решение (порог, вес, комбинация признаков)
    остаётся за моделью Участника 1, которая должна уметь объяснить, ПОЧЕМУ
    именно эти признаки и с какими весами повлияли на итоговую уверенность.

Единственные внешние зависимости всего пакета `extraction/` — на этапе
разбора текста (regex, rapidfuzz для сравнения строк, py3langid для языка) —
это инфраструктурные утилиты общего назначения (сравнение строк, определение
языка), а не алгоритм "это тренд или шум". Сам критерий "тренд vs шум"
нигде не делегирован сторонней библиотеке.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..retrieval.schemas import Candidate, ClaimType, SourceDocument, SourceType, TrustLevel

# Человекочитаемое описание каждого признака — используется в UI/отчёте
# ("почему модель уверена именно так") и в защите методологии перед жюри.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "independent_confirmations": (
        "Число независимых подтверждений (разных инфоповодов/доменов). "
        "Гипотеза: у шума и маркетингового вброса обычно 0-1 подтверждение "
        "из одного источника; у формирующегося тренда со временем появляется "
        "несколько независимых упоминаний."
    ),
    "has_academic_or_patent_evidence": (
        "Есть ли среди источников научная публикация или патент. Гипотеза: "
        "присутствие проверяемого первоисточника (не пресс-релиза) — сильный "
        "сигнал того, что за упоминанием стоит реальная разработка, а не хайп."
    ),
    "high_trust_source_count": (
        "Число документов с доверием HIGH. Больше высокодоверенных "
        "источников — меньше вероятность, что кандидат держится на рекламной "
        "публикации."
    ),
    "lowest_trust_only": (
        "Кандидат держится ИСКЛЮЧИТЕЛЬНО на источниках низкого доверия "
        "(соцсети/блоги/агрегаторы/пресс-релизы/форумы). Прямой признак "
        "'маркетингового шума' по правилам ТЗ — такой кандидат не должен "
        "попадать в топ без независимого подтверждения."
    ),
    "stage_fact_count": (
        "Число извлечённых фактов о стадии развития (раунды, пилоты, патенты, "
        "публикации). Гипотеза: у зрелой технологии таких фактов накапливается "
        "много и по разным стадиям; у истинно раннего сигнала — мало и "
        "преимущественно ранние стадии (pre-seed/pilot/patent filed)."
    ),
    "early_stage_marker_ratio": (
        "Доля стадийных фактов, относящихся к ранним стадиям (pre-seed/seed, "
        "stealth, pilot/PoC, патентная заявка) среди всех стадийных фактов. "
        "Высокое значение — характерно для слабого сигнала; низкое (много "
        "'серийное производство'/'коммерческий запуск') — признак уже "
        "сформированного рынка, который по ТЗ не должен попадать в выдачу."
    ),
    "future_looking_ratio": (
        "Доля утверждений, сформулированных как планы на будущее, а не "
        "свершившиеся факты. Слишком высокое значение (почти всё — обещания) "
        "— признак маркетингового анонса без подтверждённой реализации."
    ),
    "distinct_company_count": (
        "Число разных компаний/организаций, упомянутых в связи с кандидатом. "
        "Больше одного независимого игрока может говорить о зарождении "
        "категории, а не о PR одной компании."
    ),
    "distinct_domain_count": (
        "Число разных интернет-доменов среди источников кандидата (без учёта "
        "дублей и документов одного инфоповода)."
    ),
    "has_advantage_evidence": (
        "Есть ли хотя бы одно текстово подтверждённое преимущество технологии "
        "(а не только упоминание названия)."
    ),
    "has_case_example_evidence": (
        "Есть ли хотя бы один подтверждённый кейс-пример применения."
    ),
    "days_since_earliest_mention": (
        "Число дней от самой ранней датированной публикации о кандидате до "
        "самой поздней. Очень короткий период при большом числе подтверждений "
        "может говорить о вбросе; растянутый во времени рост числа упоминаний "
        "— о постепенно набирающем силу тренде (это же поле 'Тренд упоминаний' "
        "в исходном датасете методологов)."
    ),
}

_EARLY_STAGE_LABELS = {
    "pre-seed / seed финансирование",
    "выход из stealth",
    "пилот / прототип / PoC",
    "патентная заявка",
    "научная публикация / препринт",
}


def candidate_features(candidate: Candidate, documents: Optional[list[SourceDocument]] = None) -> dict:
    """Считает прозрачный набор признаков для одного кандидата.

    `documents` — опционально, полный список SourceDocument кандидата (для
    расчёта `distinct_domain_count`/`days_since_earliest_mention`); если не
    передан, эти два признака вычисляются по `candidate.evidence` (менее
    точно, т.к. Evidence не всегда хранит домен явно).
    """
    stage_facts = candidate.stage_facts
    stage_count = len(stage_facts)
    early_count = sum(1 for sf in stage_facts if sf.label in _EARLY_STAGE_LABELS)

    all_claims = candidate.evidence
    future_count = sum(1 for e in all_claims if e.is_future_looking)

    high_trust_docs = 0
    academic_or_patent = False
    domains: set[str] = set()
    dates: list[datetime] = []

    docs_by_id = {d.doc_id: d for d in (documents or [])}
    relevant_docs = [docs_by_id[i] for i in candidate.source_doc_ids if i in docs_by_id]

    for doc in relevant_docs:
        if doc.duplicate_of:
            continue
        if doc.trust_level == TrustLevel.HIGH:
            high_trust_docs += 1
        if doc.source_type in (SourceType.ACADEMIC, SourceType.PATENT):
            academic_or_patent = True
        if doc.domain:
            domains.add(doc.domain)
        if doc.published_at:
            dates.append(doc.published_at)

    if documents is None:
        # запасной путь: используем то, что есть в самом Evidence
        academic_or_patent = any(e.source_type in (SourceType.ACADEMIC, SourceType.PATENT) for e in all_claims)
        high_trust_docs = len({e.doc_id for e in all_claims if e.trust_level == TrustLevel.HIGH})
        dates = [e.published_at for e in all_claims if e.published_at]

    days_since_earliest = None
    if len(dates) >= 2:
        days_since_earliest = (max(dates) - min(dates)).days

    return {
        "independent_confirmations": candidate.independent_confirmations,
        "has_academic_or_patent_evidence": academic_or_patent,
        "high_trust_source_count": high_trust_docs,
        "lowest_trust_only": candidate.lowest_trust_only,
        "stage_fact_count": stage_count,
        "early_stage_marker_ratio": (early_count / stage_count) if stage_count else None,
        "future_looking_ratio": (future_count / len(all_claims)) if all_claims else None,
        "distinct_company_count": len(candidate.companies),
        "distinct_domain_count": len(domains) if domains else None,
        "has_advantage_evidence": len(candidate.advantages) > 0,
        "has_case_example_evidence": len(candidate.case_examples) > 0,
        "days_since_earliest_mention": days_since_earliest,
    }


def explain_features(features: dict) -> list[str]:
    """Возвращает список строк 'признак = значение — почему это важно',
    пригодный для прямого показа в UI ('ключевые предикторы', ТЗ п. про
    интерфейс) без обращения к автору кода.
    """
    lines = []
    for key, value in features.items():
        description = FEATURE_DESCRIPTIONS.get(key, "")
        lines.append(f"{key} = {value}. {description}")
    return lines
