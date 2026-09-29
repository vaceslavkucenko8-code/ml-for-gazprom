"""
src/retrieval/discovery.py

Режим "без заданной темы" (ТЗ, п. 2.5: "расширять поиск при недостаточном
числе подтверждённых технологий", и общее требование — сервис должен САМ
находить кандидатов в слабые сигналы, а не ждать, что пользователь заранее
знает тему).

Как это устроено, и почему это НЕ скрытый алгоритм "детекции тренда":

  1. Вместо конкретной темы от пользователя берётся заранее заданный,
     явный список широких технологических областей (`DEFAULT_SEED_TOPICS`)
     — они совпадают с колонкой "Область" исходного датасета организаторов
     (Защита ИИ, Edge, Инфраструктура ИИ, Робототехника, Финтех, Квантовые
     технологии, Материалы, Энергетика). Это ОБЛАСТИ ПОИСКА, а не список
     конкретных технологий — внутри каждой области поиск (`search.py`)
     остаётся полностью свободным, не ограниченным справочником из 100
     сигналов (ТЗ, п. 2.1).
  2. Для каждой области запускается ТОТ ЖЕ САМЫЙ конвейер, что и при ручном
     поиске по теме: `SearchOrchestrator` -> `deduplicate_and_score` ->
     `extract_candidates` -> `extraction.relevance.assess_relevance`. Ни
     одна новая "решающая" логика здесь не добавляется — решение "сигнал
     или шум" для каждого найденного кандидата всё так же принимает
     явный, объяснимый rule-based скоринг из `extraction/relevance.py`.
  3. Документы со всех областей объединяются и дедуплицируются ОДНИМ общим
     проходом (а не по отдельности на каждую область) — иначе одна и та же
     статья, случайно найденная по двум разным областям (например, "edge"
     и "робототехника"), считалась бы двумя разными подтверждениями.

Итог: пользователю достаточно вызвать `discover_trends()` без единого
аргумента, чтобы получить готовый, ранжированный и объяснённый список
кандидатов — искать тему самостоятельно не нужно.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from ..extraction.candidates import extract_candidates
from ..extraction.relevance import RelevanceAssessment, RelevanceVerdict, assess_candidates
from .deduplicate import deduplicate_and_score
from .schemas import Candidate, SourceDocument
from .search import ConnectorRunError, SearchOrchestrator

DISCOVERY_QUERY_LABEL = "автоматическое обнаружение (без заданной темы)"

# Широкие области поиска по умолчанию — совпадают с колонкой "Область" из
# стартового датасета организаторов, НЕ являются справочником конкретных
# технологий (внутри каждой области поиск свободный, см. docstring выше).
# Можно передать свой словарь {метка_области: тема} в discover_trends().
#
# Намеренно без общих слов-заполнителей вроде "новые"/"технологии"/"решения":
# живой прогон показал реальный дефект — если КАЖДАЯ тема содержит одно и то
# же общее слово, лексический фильтр релевантности в search.py
# (_matches_topic) пропускает по нему вообще любой документ с этим словом,
# независимо от области (например, "Новые технологии реализации
# межбюджетных отношений" проходил фильтр только из-за слова "новые"/
# "технологии"). Ниже — только различающие, содержательные существительные
# самой области.
DEFAULT_SEED_TOPICS: dict[str, str] = {
    "Защита ИИ": "защита ИИ-моделей от атак red teaming уязвимости ИИ-агентов",
    "Edge": "edge-вычисления периферийные вычисления на устройстве IoT",
    "Инфраструктура ИИ": "чипы дата-центры инференс обучение ИИ-моделей ускорители",
    "Робототехника": "автономные роботы манипуляторы робототехника",
    "Финтех": "платёжные технологии финтех криптовалюта блокчейн",
    "Квантовые технологии": "квантовые вычисления квантовые сенсоры квантовая криптография",
    "Материалы": "наноматериалы умные материалы сенсоры",
    "Энергетика": "накопители энергии батареи возобновляемая энергетика",
}


@dataclasses.dataclass
class DiscoveryResult:
    """Результат автоматического обхода нескольких широких областей без
    заданной пользователем темы."""

    seed_topics: dict[str, str]
    documents: list[SourceDocument] = dataclasses.field(default_factory=list)
    candidates: list[Candidate] = dataclasses.field(default_factory=list)
    assessments: dict[str, RelevanceAssessment] = dataclasses.field(default_factory=dict)
    errors: list[ConnectorRunError] = dataclasses.field(default_factory=list)
    connectors_used: list[str] = dataclasses.field(default_factory=list)
    connectors_skipped: list[str] = dataclasses.field(default_factory=list)
    filtered_low_relevance: int = 0

    def ranked_candidates(self, *, drop_noise: bool = True) -> list[Candidate]:
        """Кандидаты по убыванию score (extraction/relevance.py). По
        умолчанию без вердикта `likely_noise` — то есть уже готовый,
        отфильтрованный список "вероятных сигналов", а не сырой список
        всего найденного (искать/отбирать вручную не нужно).
        """
        cands = self.candidates
        if drop_noise:
            cands = [c for c in cands if self.assessments[c.candidate_id].verdict != RelevanceVerdict.LIKELY_NOISE]
        return sorted(cands, key=lambda c: self.assessments[c.candidate_id].score, reverse=True)

    def summary(self) -> dict:
        return {
            "seed_topics": self.seed_topics,
            "documents_found": len(self.documents),
            "candidates_found": len(self.candidates),
            "filtered_low_relevance": self.filtered_low_relevance,
            "connectors_used": sorted(set(self.connectors_used)),
            "connectors_skipped": sorted(set(self.connectors_skipped)),
            "errors": [dataclasses.asdict(e) for e in self.errors],
            "verdict_counts": {
                verdict.value: sum(1 for a in self.assessments.values() if a.verdict == verdict)
                for verdict in RelevanceVerdict
            },
        }


def discover_trends(
    seed_topics: Optional[dict[str, str]] = None,
    *,
    orchestrator: Optional[SearchOrchestrator] = None,
    max_queries_per_topic: int = 2,
) -> DiscoveryResult:
    """Обходит все `seed_topics` (по умолчанию `DEFAULT_SEED_TOPICS`) без
    какой-либо темы от пользователя, находит документы, дедуплицирует их
    ОДНИМ общим проходом, извлекает кандидатов и оценивает каждого через
    `extraction.relevance.assess_relevance` — тем же кодом, что и при
    ручном поиске по конкретной теме.
    """
    seed_topics = seed_topics or DEFAULT_SEED_TOPICS
    orchestrator = orchestrator or SearchOrchestrator()

    result = DiscoveryResult(seed_topics=seed_topics)
    seen_urls: set[str] = set()
    all_documents: list[SourceDocument] = []

    for topic in seed_topics.values():
        run = orchestrator.run(topic, max_queries=max_queries_per_topic)
        result.errors.extend(run.errors)
        result.connectors_used.extend(run.connectors_used)
        result.connectors_skipped.extend(run.connectors_skipped)
        result.filtered_low_relevance += run.filtered_low_relevance

        for doc in run.documents:
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            all_documents.append(doc)

    documents = deduplicate_and_score(all_documents)
    result.documents = documents

    candidates = extract_candidates(DISCOVERY_QUERY_LABEL, documents)
    result.candidates = candidates
    result.assessments = {a.candidate_id: a for a in assess_candidates(candidates, documents)}

    return result
