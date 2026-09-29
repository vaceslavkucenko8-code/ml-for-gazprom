"""src/continuous_ingestion.py — self-growing corpus, built ONLY by wiring
together already-existing, already-tested algorithms (search, dedup,
extraction, relevance rule, automatic evidence rule). No parallel scoring
logic is introduced here; these tests check the wiring and the storage
invariants, not the rule engines themselves (see test_discovery.py /
test_automatic.py for those)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continuous_ingestion import _adaptive_topics, run_cycle, run_loop
from src.corpus_store import CorpusStore
from src.retrieval.connectors.base import SearchConnector, SearchHit
from src.retrieval.search import SearchOrchestrator


class _FakeAcademicConnector(SearchConnector):
    """Two independent domains confirm the same claim — enough for the
    relevance rule to score above the noise threshold (mirrors the fixture
    already validated in tests/participant2/test_discovery.py)."""

    name = "fake_academic"
    description = "фиктивный академический коннектор для теста автосбора"
    default_source_type = "academic_paper"

    def search(self, query, max_results=10):
        if "квантовые" not in query:
            return []
        return [
            SearchHit(
                url="https://example.org/quantum-sensor",
                title="Новые квантовые технологии: пилотное испытание сенсора",
                full_text=(
                    "Компания провела пилотное испытание сенсора в рамках направления "
                    "новые квантовые технологии, патентная заявка подана."
                ),
                connector=self.name, query=query,
            ),
            SearchHit(
                url="https://research.example/quantum-sensor-confirmation",
                title="Независимое подтверждение: новые квантовые технологии сенсора",
                full_text=(
                    "Другая лаборатория независимо подтвердила новые квантовые технологии "
                    "в области сенсоров: прототип испытан повторно."
                ),
                connector=self.name, query=query,
            ),
        ]


class _FakeBlogConnector(SearchConnector):
    """Single low-trust source, no independent confirmation — guaranteed
    likely_noise verdict, same fixture shape as test_discovery.py."""

    name = "fake_blog"
    description = "фиктивный блог-коннектор для теста автосбора"
    default_source_type = "personal_blog"

    def search(self, query, max_results=10):
        if "робот" not in query:
            return []
        return [
            SearchHit(
                url="https://blog.example/robot-noise",
                title="Личный блог про новые автономные роботы",
                full_text="Кто-то написал в личном блоге про новые автономные роботы без единого подтверждения.",
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                connector=self.name, query=query,
            ),
        ]


class _EmptyConnector(SearchConnector):
    name = "fake_empty"
    description = "не находит ничего — для быстрых тестов планировщика"
    default_source_type = "unknown"

    def search(self, query, max_results=10):
        return []


QUANTUM_SEEDS = {"Квантовые технологии": "новые квантовые технологии"}
ROBOTICS_SEEDS = {"Робототехника": "новые автономные роботы"}


def _quantum_orchestrator():
    return SearchOrchestrator(connectors=[_FakeAcademicConnector()], fetch_full_pages=False)


def _robotics_orchestrator():
    return SearchOrchestrator(connectors=[_FakeBlogConnector()], fetch_full_pages=False)


def test_run_cycle_persists_documents_and_both_assessments(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    report = run_cycle(
        store, _quantum_orchestrator(), seed_topics=QUANTUM_SEEDS,
        max_seed_topics=1, max_adaptive_topics=0, max_queries_per_topic=1,
    )
    assert report["status"] == "ok"
    assert report["counts"]["new_documents"] == 2
    assert report["counts"]["new_candidates"] >= 1

    stats = store.stats()
    assert stats["documents"] == 2
    assert stats["candidates"] >= 1

    rows = store.candidates()
    for row in rows:
        payload = json.loads(row["payload_json"])
        # both independent, already-existing rule engines must be attached
        assert "relevance_verdict" in payload and payload["relevance_verdict"]
        assert "automatic_decision" in payload  # may be None only if no candidates at all
        assert payload["label_kind"] == "automatic_hypothesis_not_expert_ground_truth"
        assert bool(row["included"]) == (payload["relevance_verdict"] != "likely_noise")
    store.close()


def test_run_cycle_excludes_likely_noise_candidate(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    report = run_cycle(
        store, _robotics_orchestrator(), seed_topics=ROBOTICS_SEEDS,
        max_seed_topics=1, max_adaptive_topics=0, max_queries_per_topic=1,
    )
    assert report["status"] == "ok"
    rows = store.candidates()
    assert len(rows) == 1
    assert rows[0]["relevance_verdict"] == "likely_noise"
    assert rows[0]["included"] == 0
    assert rows[0]["exclude_reason"] == "likely_noise_relevance_rule"
    store.close()


def test_run_cycle_does_not_reprocess_already_known_documents(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    first = run_cycle(
        store, _quantum_orchestrator(), seed_topics=QUANTUM_SEEDS,
        max_seed_topics=1, max_adaptive_topics=0, max_queries_per_topic=1,
    )
    candidates_after_first = store.stats()["candidates"]

    second = run_cycle(
        store, _quantum_orchestrator(), seed_topics=QUANTUM_SEEDS,
        max_seed_topics=1, max_adaptive_topics=0, max_queries_per_topic=1,
    )
    assert second["status"] == "ok"
    assert second["counts"]["new_documents"] == 0
    assert second["counts"]["new_candidates"] == 0
    assert store.stats()["documents"] == 2  # unchanged
    assert store.stats()["candidates"] == candidates_after_first  # unchanged
    store.close()


def test_run_cycle_never_writes_outside_the_given_corpus_db(tmp_path):
    db_path = tmp_path / "isolated" / "corpus.db"
    store = CorpusStore(db_path)
    run_cycle(store, _quantum_orchestrator(), seed_topics=QUANTUM_SEEDS, max_seed_topics=1, max_adaptive_topics=0)
    assert db_path.exists()
    # nothing else should have been created next to it
    assert list(db_path.parent.iterdir()) == [db_path]
    store.close()


def test_run_cycle_records_error_status_without_raising_when_storage_breaks(tmp_path, monkeypatch):
    store = CorpusStore(tmp_path / "corpus.db")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(store, "upsert_document", _boom)
    report = run_cycle(store, _quantum_orchestrator(), seed_topics=QUANTUM_SEEDS, max_seed_topics=1, max_adaptive_topics=0)
    assert report["status"] == "error"
    assert "simulated storage failure" in report["error"]
    cycles = store.recent_cycles()
    assert cycles[0]["status"] == "error"
    store.close()


def test_adaptive_topics_are_derived_from_best_scoring_candidate_and_marked_used(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_candidate("cycle_1", {
        "candidate_id": "cand_high", "technology_name": "Гидрофон MEMS", "domain": "Материалы",
        "seed_topic": "Материалы", "query": "тест", "relevance_score": 0.9,
        "relevance_verdict": "likely_emerging_signal", "is_actual": True,
        "automatic_decision": "needs_review", "automatic_reason": "insufficient_explicit_evidence",
        "automatic_score": 30, "included": True, "exclude_reason": None,
    })
    topics = _adaptive_topics(store, limit=2)
    assert len(topics) == 1
    (label, topic), = topics.items()
    assert label.startswith("Адаптивно:")
    assert "Гидрофон MEMS" in topic

    # once used, the same candidate must not spawn a second adaptive topic
    assert _adaptive_topics(store, limit=2) == {}
    store.close()


def test_run_loop_respects_max_cycles_and_writes_heartbeat(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    sleeps = []
    reports = run_loop(
        store,
        orchestrator_factory=lambda: SearchOrchestrator(connectors=[_EmptyConnector()], fetch_full_pages=False),
        interval_seconds=1234,
        max_cycles=2,
        seed_topics={"A": "тема а", "B": "тема б"},
        max_seed_topics=1,
        max_adaptive_topics=0,
        sleep_fn=sleeps.append,
        heartbeat_path=tmp_path / "heartbeat.json",
    )
    assert len(reports) == 2
    assert all(r["status"] == "ok" for r in reports)
    # scheduler must not sleep after the final cycle it was asked to run
    assert sleeps == [1234]

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["cycle_number"] == 2
    assert heartbeat["interval_seconds"] == 1234
    store.close()
