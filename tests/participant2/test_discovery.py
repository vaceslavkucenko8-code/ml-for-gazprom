import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.relevance import RelevanceVerdict
from src.retrieval.connectors.base import SearchConnector, SearchHit
from src.retrieval.discovery import DEFAULT_SEED_TOPICS, discover_trends
from src.retrieval.search import SearchOrchestrator

# Тексты ниже намеренно содержат слова темы буквально (а не в другом падеже)
# — SearchOrchestrator.run() пропускает находки через лексический фильтр
# релевантности (search._matches_topic), который сравнивает точные подстроки
# со словами исходной темы, без учёта словоформ. Это тот же самый фильтр,
# что работает при живом поиске, поэтому фикстуры должны его проходить.


class _FakeAcademicConnector(SearchConnector):
    """Возвращает разные документы для разных seed-тем — имитирует то, что
    в реальности делают arXiv/Crossref для разных широких областей поиска.
    Один URL повторяется под двумя темами, чтобы проверить сквозной
    (across-topics) дедуп по URL.
    """

    name = "fake_academic"
    description = "фиктивный академический коннектор для теста автообнаружения"
    default_source_type = "academic_paper"

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        if "квантовые" in query:
            return [
                SearchHit(
                    url="https://example.org/quantum-sensor",
                    title="Новые квантовые технологии: пилотное испытание сенсора",
                    full_text=(
                        "Компания провела пилотное испытание сенсора в рамках "
                        "направления новые квантовые технологии, патентная заявка подана."
                    ),
                    connector=self.name,
                    query=query,
                ),
                # тот же URL встречается и под темой "материалы" — должен
                # быть посчитан только один раз на весь discovery-прогон
                SearchHit(
                    url="https://example.org/shared-across-topics",
                    title="Материалы для новые квантовые технологии сенсоров",
                    full_text="Исследование материала: новые квантовые технологии и сенсоры применяются совместно.",
                    connector=self.name,
                    query=query,
                ),
            ]
        if "материалы" in query or "сенсоры" in query:
            return [
                SearchHit(
                    url="https://example.org/shared-across-topics",
                    title="Материалы для новые квантовые технологии сенсоров",
                    full_text="Исследование материала: новые материалы и сенсоры применяются совместно.",
                    connector=self.name,
                    query=query,
                ),
            ]
        return []


class _FakeBlogConnector(SearchConnector):
    """Отдельный коннектор с default_source_type="personal_blog" —
    имитирует низкодоверенный источник без единого подтверждения, чтобы
    проверить, что ranked_candidates(drop_noise=True) действительно
    отфильтровывает вердикт LIKELY_NOISE.
    """

    name = "fake_blog"
    description = "фиктивный блог-коннектор для теста автообнаружения"
    default_source_type = "personal_blog"

    def search(self, query: str, max_results: int = 10) -> list[SearchHit]:
        if "робот" in query:
            return [
                SearchHit(
                    url="https://blog.example/robot-noise",
                    title="Личный блог про новые автономные роботы",
                    full_text="Кто-то написал в личном блоге про новые автономные роботы без единого подтверждения.",
                    published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    connector=self.name,
                    query=query,
                ),
            ]
        return []


def test_discover_trends_works_without_any_user_supplied_topic():
    orchestrator = SearchOrchestrator(
        connectors=[_FakeAcademicConnector(), _FakeBlogConnector()], fetch_full_pages=False
    )
    seeds = {
        "Квантовые технологии": "новые квантовые технологии",
        "Материалы": "новые материалы и сенсоры",
        "Робототехника": "новые автономные роботы",
    }
    result = discover_trends(seeds, orchestrator=orchestrator, max_queries_per_topic=1)

    assert len(result.documents) > 0
    assert len(result.candidates) > 0
    assert set(result.assessments.keys()) == {c.candidate_id for c in result.candidates}


def test_discover_trends_deduplicates_same_url_across_different_seed_topics():
    orchestrator = SearchOrchestrator(connectors=[_FakeAcademicConnector()], fetch_full_pages=False)
    seeds = {
        "Квантовые технологии": "новые квантовые технологии",
        "Материалы": "новые материалы и сенсоры",
    }
    result = discover_trends(seeds, orchestrator=orchestrator, max_queries_per_topic=1)

    urls = [d.url for d in result.documents]
    assert urls.count("https://example.org/shared-across-topics") == 1


def test_ranked_candidates_drops_likely_noise_by_default():
    orchestrator = SearchOrchestrator(connectors=[_FakeBlogConnector()], fetch_full_pages=False)
    seeds = {"Робототехника": "новые автономные роботы"}
    result = discover_trends(seeds, orchestrator=orchestrator, max_queries_per_topic=1)

    assert len(result.candidates) == 1
    only_id = result.candidates[0].candidate_id
    assert result.assessments[only_id].verdict == RelevanceVerdict.LIKELY_NOISE

    ranked_ids_with_noise = {c.candidate_id for c in result.ranked_candidates(drop_noise=False)}
    ranked_ids_without_noise = {c.candidate_id for c in result.ranked_candidates(drop_noise=True)}

    assert ranked_ids_with_noise == {only_id}
    assert ranked_ids_without_noise == set()


def test_summary_reports_verdict_counts_and_seed_topics():
    orchestrator = SearchOrchestrator(connectors=[_FakeAcademicConnector()], fetch_full_pages=False)
    seeds = {"Квантовые технологии": "новые квантовые технологии"}
    result = discover_trends(seeds, orchestrator=orchestrator, max_queries_per_topic=1)
    summary = result.summary()

    assert summary["seed_topics"] == seeds
    assert sum(summary["verdict_counts"].values()) == len(result.candidates)
    assert summary["documents_found"] == len(result.documents)


def test_discover_trends_defaults_to_default_seed_topics_when_none_given():
    orchestrator = SearchOrchestrator(connectors=[_FakeAcademicConnector()], fetch_full_pages=False)
    result = discover_trends(orchestrator=orchestrator, max_queries_per_topic=1)
    assert result.seed_topics == DEFAULT_SEED_TOPICS


def test_default_seed_topics_do_not_share_a_common_filler_word():
    # Регрессионный тест на реальный дефект живого прогона: если КАЖДАЯ
    # seed-тема содержит одно и то же общее слово (например, "новые"), то
    # лексический фильтр релевантности search._matches_topic пропускает по
    # этому слову вообще любой документ независимо от области — фильтр
    # молча перестаёт что-либо отличать.
    from src.retrieval.search import _topic_keywords

    keyword_sets = [set(_topic_keywords(t)) for t in DEFAULT_SEED_TOPICS.values()]
    common_to_all = set.intersection(*keyword_sets)
    assert not common_to_all, (
        f"слово(а) {common_to_all} встречаются во ВСЕХ seed-темах — "
        "фильтр релевантности не сможет отличить область по этому слову"
    )
