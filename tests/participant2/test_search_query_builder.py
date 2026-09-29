import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.retrieval.connectors.base import SearchConnector, SearchHit
from src.retrieval.search import SearchOrchestrator, build_queries


def test_build_queries_from_free_topic_not_in_any_fixed_list():
    # тема заведомо не входит ни в какой справочник — модуль не должен
    # завязываться на конечный список технологий
    queries = build_queries("биопечать органов на чипе, минуя донорские листы ожидания")
    assert len(queries) >= 2
    assert all("биопечать" in q for q in queries)


def test_build_queries_english_topic_uses_english_modifiers():
    queries = build_queries("photonic neuromorphic accelerators")
    assert any("startup" in q or "pilot" in q or "preprint" in q or "patent" in q for q in queries)


def test_build_queries_empty_raises():
    with pytest.raises(ValueError):
        build_queries("   ")


def test_build_queries_deduplicates_variants():
    queries = build_queries("тема", max_variants=10)
    assert len(queries) == len(set(queries))


class _FakeSeedConnector(SearchConnector):
    """Имитирует реальный обнаруженный дефект: коннектор, который у запроса
    '<тема> стартап pre-seed seed' матчится буквально по слову 'seed' и
    возвращает документы, никак не связанные с исходной темой.
    """

    name = "fake_seed"
    description = "фиктивный коннектор для теста фильтра релевантности"
    default_source_type = "academic_paper"

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        # Current short modifiers can produce the same topic-drift failure.
        if "seed" in query or "прототип" in query:
            return [SearchHit(
                url="https://example.org/seed-coder",
                title="Seed-Coder: Let the Code Model Curate Data for Itself",
                full_text="A paper about curating training data for code models, unrelated to the search topic.",
                connector=self.name,
                query=query,
            )]
        return [SearchHit(
            url="https://example.org/quantum-navigation",
            title="Квантовые сенсоры для навигации без GPS: пилотный проект",
            full_text="Пилотный проект квантового сенсора для навигации без GPS показал точность в пределах метра.",
            connector=self.name,
            query=query,
        )]


def test_orchestrator_filters_out_hits_unrelated_to_original_topic():
    orchestrator = SearchOrchestrator(connectors=[_FakeSeedConnector()], fetch_full_pages=False)
    run = orchestrator.run("квантовые сенсоры для навигации без GPS", max_queries=4)

    urls = {d.url for d in run.documents}
    assert "https://example.org/seed-coder" not in urls
    assert "https://example.org/quantum-navigation" in urls
    assert run.filtered_low_relevance >= 1
    assert run.summary()["filtered_low_relevance"] == run.filtered_low_relevance
