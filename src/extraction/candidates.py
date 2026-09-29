"""
src/extraction/candidates.py

Реализует п. 2.4 (и частично 2.5) ТЗ: из найденных и обработанных документов
выделяются кандидаты в слабые сигналы — название технологии, краткое
описание, область, компании, факты о стадии развития, преимущества,
кейс-примеры — каждый факт со ссылкой на документ и подтверждающую цитату.

Явно НЕ входит в задачу этого модуля: финальная классификация "это слабый
сигнал / не слабый сигнал" и итоговый скоринг — это модель Участника 1.
`Candidate` (см. schemas.py) физически не имеет такого поля.

Архитектура извлечения:
  1. `cluster_documents_into_candidates` — группирует документы по теме
     (техническая сущность/компания), используя нечёткое сравнение заголовков
     и общий поисковый запрос. Один кластер = один Candidate.
  2. `RuleBasedExtractor` — для каждого документа в кластере вытаскивает
     предложения-кандидаты в Evidence по словарям из keywords.py: стадия
     развития, преимущество, кейс-пример; определяет "прошедшее
     событие/будущий план" по маркерам времени.
  3. `CandidateExtractor` — абстрактный интерфейс, чтобы в будущем можно было
     подключить LLM-based экстрактор (через approved cloud API/локальную
     модель из ТЗ, п. 3.1) без изменения остального пайплайна — нужно только
     реализовать тот же интерфейс.
"""

from __future__ import annotations

import abc
import logging
import re
from collections import defaultdict
from typing import Optional

from rapidfuzz import fuzz

from ..retrieval.deduplicate import count_independent_confirmations, LOW_TRUST_TYPES
from ..retrieval.schemas import (
    Advantage,
    CaseExample,
    Candidate,
    ClaimType,
    DevelopmentStageFact,
    Evidence,
    SourceDocument,
)
from .keywords import (
    ADVANTAGE_MARKERS_EN,
    ADVANTAGE_MARKERS_RU,
    CASE_MARKERS_EN,
    CASE_MARKERS_RU,
    FUTURE_MARKERS_EN,
    FUTURE_MARKERS_RU,
    STAGE_MARKERS,
    guess_domain,
)

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z])")

# Организационные суффиксы/паттерны, используемые эвристикой извлечения компаний
_ORG_SUFFIXES = (
    "Inc", "Inc.", "LLC", "Ltd", "Ltd.", "Labs", "AI", "Systems", "Technologies",
    "Corp", "Corp.", "GmbH", "Robotics", "Security", "Networks",
)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Preserve whitespace inside quotes for exact source matching.
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def is_future_looking(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(re.search(r'\b(?:could|would|might|may|will)\b', lowered)) or any(m in lowered for m in FUTURE_MARKERS_RU) or any(m in lowered for m in FUTURE_MARKERS_EN)


def _find_matches(sentence: str, marker_lists: list[list[str]]) -> bool:
    lowered = sentence.lower()
    return any(any(m in lowered for m in markers) for markers in marker_lists)


def extract_companies(text: str, title: str = "") -> list[str]:
    """Грубая эвристика извлечения названий компаний: последовательности слов
    с заглавной буквы, оканчивающиеся на типичный организационный суффикс,
    плюс эвристика "Заглавное Слово" в начале заголовка (частый паттерн
    новостных заголовков вида 'CompanyName Raises $X').
    """
    found: set[str] = set()
    for suffix in _ORG_SUFFIXES:
        pattern = re.compile(
            r"\b([A-ZА-Я][\w&.\-]*(?:\s+[A-ZА-Я][\w&.\-]*){0,3}\s+" + re.escape(suffix) + r")\b"
        )
        for m in pattern.finditer(text + " " + title):
            found.add(m.group(1).strip())

    # частый паттерн: "<Компания> Raises $X" / "<Компания> привлекла $X"
    m = re.match(r"^([A-ZА-Я][\w.\-]*(?:\s+[A-ZА-Я][\w.\-]*){0,2})\s+(?:Raises|привлекл|запустил|представил)", title)
    if m:
        found.add(m.group(1).strip())

    return sorted(found)[:6]


# --------------------------------------------------------------------------- #
# Кластеризация документов в кандидатов
# --------------------------------------------------------------------------- #

CLUSTER_TITLE_SIMILARITY_THRESHOLD = 85


def cluster_documents_into_candidates(documents: list[SourceDocument]) -> list[list[SourceDocument]]:
    """Группирует документы по предполагаемой технологии/компании внутри
    одной поисковой темы. Кластеризация — по нечёткому сравнению заголовков
    (rapidfuzz) плюс общий набор извлечённых названий компаний.

    Это НАМЕРЕННО простая эвристика: у Участника 2 нет доступа к финальной
    модели классификации, и на этом этапе достаточно правдоподобной
    группировки "один кандидат = документы об одном и том же продукте/
    компании/технологии", чтобы передать дальше проверяемые факты.
    """
    clusters: list[list[SourceDocument]] = []
    cluster_keys: list[str] = []  # представительный заголовок кластера
    from ..source_provenance import work_id

    def own_ids(doc):
        raw = doc.raw_metadata
        return {key for value in (doc.url, raw.get('doi'), raw.get('meta', {}).get('citation_doi'))
                if (key := work_id(value))}

    for doc in documents:
        if doc.duplicate_of:
            continue  # дубли не создают отдельных кандидатов
        title = doc.title or doc.text[:80] or doc.url
        companies = extract_companies(doc.text or "", title)
        key_text = (title + " " + " ".join(companies)).strip()

        identity = own_ids(doc)
        same_work = next((i for i, cluster in enumerate(clusters)
                          if any(identity & own_ids(other) for other in cluster)), None)
        if same_work is not None:
            clusters[same_work].append(doc)
            continue

        best_idx, best_score = None, 0
        for idx, key in enumerate(cluster_keys):
            score = fuzz.token_sort_ratio(key_text.casefold(), key.casefold())
            if score > best_score:
                best_idx, best_score = idx, score

        if best_idx is not None and best_score >= CLUSTER_TITLE_SIMILARITY_THRESHOLD:
            clusters[best_idx].append(doc)
        else:
            clusters.append([doc])
            cluster_keys.append(key_text)

    return clusters


# --------------------------------------------------------------------------- #
# Экстрактор
# --------------------------------------------------------------------------- #


class CandidateExtractor(abc.ABC):
    """Интерфейс извлечения кандидатов. Реализация по умолчанию —
    RuleBasedExtractor; точка расширения для LLM-based извлечения.
    """

    @abc.abstractmethod
    def extract(self, query: str, documents: list[SourceDocument]) -> list[Candidate]:
        raise NotImplementedError


class RuleBasedExtractor(CandidateExtractor):
    def extract(self, query: str, documents: list[SourceDocument]) -> list[Candidate]:
        clusters = cluster_documents_into_candidates(documents)
        candidates: list[Candidate] = []

        for cluster in clusters:
            candidate = self._build_candidate(query, cluster)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _build_candidate(self, query: str, docs: list[SourceDocument]) -> Optional[Candidate]:
        if not docs:
            return None

        primary_doc = max(docs, key=lambda d: len(d.text or ""))
        technology_name = primary_doc.title or primary_doc.text[:60] or query
        companies: set[str] = set()
        stage_facts: list[DevelopmentStageFact] = []
        advantages: list[Advantage] = []
        case_examples: list[CaseExample] = []
        evidence: list[Evidence] = []

        for doc in docs:
            companies.update(extract_companies(doc.text or "", doc.title or ""))
            doc_evidence = self._extract_evidence_from_doc(doc)
            evidence.extend(doc_evidence)

            for ev in doc_evidence:
                if ev.claim_type == ClaimType.STAGE_OF_DEVELOPMENT:
                    stage_facts.append(
                        DevelopmentStageFact(
                            label=ev.context or "стадия развития",
                            description=ev.quote,
                            is_future_looking=ev.is_future_looking,
                            evidence_ids=[ev.evidence_id],
                        )
                    )
                elif ev.claim_type == ClaimType.ADVANTAGE:
                    advantages.append(Advantage(text=ev.quote, evidence_ids=[ev.evidence_id]))
                elif ev.claim_type == ClaimType.CASE_EXAMPLE:
                    case_examples.append(
                        CaseExample(text=ev.quote, organization=None, evidence_ids=[ev.evidence_id])
                    )

        if not evidence:
            # даже без "ярких" маркеров у кандидата должен быть хотя бы один
            # факт общего типа, иначе он бесполезен для Участника 1
            fallback_sentences = split_sentences(primary_doc.text)[:1]
            for s in fallback_sentences:
                evidence.append(
                    Evidence(
                        doc_id=primary_doc.doc_id,
                        url=primary_doc.url,
                        quote=s,
                        claim_type=ClaimType.GENERAL_FACT,
                        is_future_looking=is_future_looking(s),
                        published_at=primary_doc.published_at,
                        source_type=primary_doc.source_type,
                        trust_level=primary_doc.trust_level,
                        is_translated=primary_doc.is_translated,
                        is_generated_summary=primary_doc.is_generated_summary,
                    )
                )

        confirmation = count_independent_confirmations(docs)
        lowest_trust_only = all(d.source_type in LOW_TRUST_TYPES for d in docs if not d.duplicate_of)

        short_description = (primary_doc.text_excerpt or primary_doc.text[:300] or "").strip()

        candidate = Candidate(
            technology_name=technology_name.strip()[:200],
            short_description=short_description,
            domain=guess_domain((primary_doc.title or "") + " " + (primary_doc.text or "")),
            companies=sorted(companies)[:6],
            query=query,
            stage_facts=stage_facts[:8],
            advantages=advantages[:8],
            case_examples=case_examples[:8],
            evidence=evidence,
            source_doc_ids=[d.doc_id for d in docs],
            independent_confirmations=confirmation.independent_confirmations,
            confirmation_explanation=confirmation.explanation,
            lowest_trust_only=lowest_trust_only,
            trust_summary=self._trust_summary(docs),
            notes=(
                "Держится только на источниках низкого доверия (соцсети/блоги/агрегаторы/"
                "пресс-релизы/форумы) — по политике ТЗ не может быть единственным основанием "
                "для включения в итоговую выдачу без независимого подтверждения."
                if lowest_trust_only else None
            ),
        )
        return candidate

    def _extract_evidence_from_doc(self, doc: SourceDocument) -> list[Evidence]:
        evidence: list[Evidence] = []
        sentences = split_sentences(doc.text)

        for sentence in sentences:
            claim_type = None
            context_label = None

            for label, markers in STAGE_MARKERS.items():
                if _find_matches(sentence, [markers]):
                    claim_type = ClaimType.STAGE_OF_DEVELOPMENT
                    context_label = label
                    break

            if claim_type is None and _find_matches(sentence, [ADVANTAGE_MARKERS_RU, ADVANTAGE_MARKERS_EN]):
                claim_type = ClaimType.ADVANTAGE

            if claim_type is None and _find_matches(sentence, [CASE_MARKERS_RU, CASE_MARKERS_EN]):
                claim_type = ClaimType.CASE_EXAMPLE

            if claim_type is None:
                continue

            evidence.append(
                Evidence(
                    doc_id=doc.doc_id,
                    url=doc.url,
                    quote=sentence,
                    context=context_label,
                    claim_type=claim_type,
                    is_future_looking=is_future_looking(sentence),
                    published_at=doc.published_at,
                    source_type=doc.source_type,
                    trust_level=doc.trust_level,
                    is_translated=doc.is_translated,
                    is_generated_summary=doc.is_generated_summary,
                )
            )

        return evidence[:10]

    @staticmethod
    def _trust_summary(docs: list[SourceDocument]) -> str:
        counts: dict[str, int] = defaultdict(int)
        for d in docs:
            if d.duplicate_of:
                continue
            level = d.trust_level.value if d.trust_level else "не определён"
            counts[level] += 1
        parts = [f"{level}: {n}" for level, n in sorted(counts.items())]
        return "; ".join(parts) if parts else "нет данных о доверии"


def extract_candidates(query: str, documents: list[SourceDocument], extractor: Optional[CandidateExtractor] = None) -> list[Candidate]:
    extractor = extractor or RuleBasedExtractor()
    return extractor.extract(query, documents)
