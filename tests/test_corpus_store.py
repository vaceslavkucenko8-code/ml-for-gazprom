"""src/corpus_store.py — persistent, self-growing corpus, kept separate from
data/raw/signals.xlsx and data/self_review/labels.json (see AGENTS.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus_store import CorpusStore, DEFAULT_DB_PATH


def _doc(doc_id="doc_1", url="https://example.org/a", **extra):
    return {
        "doc_id": doc_id, "url": url, "normalized_url": url,
        "title": "Заголовок", "domain": "example.org", "source_type": "academic_paper",
        "connector": "fake", "published_at": "2026-01-01T00:00:00+00:00",
        "fetch_status": "ok", "trust_level": "high", **extra,
    }


def _candidate(cid="cand_1", **extra):
    return {
        "candidate_id": cid, "technology_name": "Тестовая технология", "domain": "Материалы",
        "seed_topic": "Материалы", "query": "тест", "relevance_score": 0.7,
        "relevance_verdict": "likely_emerging_signal", "is_actual": True,
        "automatic_decision": "needs_review", "automatic_reason": "insufficient_explicit_evidence",
        "automatic_score": 30, "included": True, "exclude_reason": None, **extra,
    }


def test_default_db_path_is_isolated_from_organizer_data_and_self_review():
    parts = DEFAULT_DB_PATH.parts
    assert "corpus" in parts
    assert "raw" not in parts
    assert "self_review" not in parts


def test_upsert_document_reports_new_then_not_new(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    assert store.upsert_document("cycle_1", _doc()) is True
    assert store.upsert_document("cycle_2", _doc()) is False  # same normalized_url
    assert store.stats()["documents"] == 1
    store.close()


def test_known_normalized_urls_round_trips(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_document("cycle_1", _doc(doc_id="doc_1", url="https://example.org/a"))
    store.upsert_document("cycle_1", _doc(doc_id="doc_2", url="https://example.org/b", normalized_url="https://example.org/b"))
    assert store.known_normalized_urls() == {"https://example.org/a", "https://example.org/b"}
    store.close()


def test_upsert_candidate_new_vs_seen_again_increments_times_seen(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    assert store.upsert_candidate("cycle_1", _candidate()) is True
    assert store.upsert_candidate("cycle_2", _candidate()) is False
    row = store.candidates()[0]
    assert row["times_seen"] == 2
    assert row["last_seen_cycle"] == "cycle_2"
    store.close()


def test_candidates_included_only_filters_excluded(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_candidate("cycle_1", _candidate(cid="cand_in", included=True))
    store.upsert_candidate("cycle_1", _candidate(cid="cand_out", included=False, exclude_reason="likely_noise_relevance_rule"))
    included = store.candidates(included_only=True)
    assert {r["candidate_id"] for r in included} == {"cand_in"}
    store.close()


def test_least_recently_queried_labels_rotates(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    labels = ["A", "B", "C", "D"]
    first_pick = store.least_recently_queried_labels(labels, 2)
    assert len(first_pick) == 2
    store.mark_labels_queried(first_pick, "cycle_1")
    second_pick = store.least_recently_queried_labels(labels, 2)
    # the two just-queried labels must not be picked again while unqueried ones remain
    assert set(second_pick).isdisjoint(set(first_pick))
    store.close()


def test_high_score_candidates_for_adaptive_seeding_excludes_already_used(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_candidate("cycle_1", _candidate(cid="cand_high", relevance_score=0.9))
    store.upsert_candidate("cycle_1", _candidate(cid="cand_low", relevance_score=0.3))
    picks = store.high_score_candidates_for_adaptive_seeding(limit=5)
    assert [r["candidate_id"] for r in picks] == ["cand_high", "cand_low"]
    store.mark_adaptive_topic_used("cand_high", "тестовая тема развитие", "Адаптивно: Тестовая технология")
    picks_after = store.high_score_candidates_for_adaptive_seeding(limit=5)
    assert [r["candidate_id"] for r in picks_after] == ["cand_low"]
    store.close()


def test_stats_never_treats_automatic_decision_as_verified_label(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_candidate("cycle_1", _candidate())
    stats = store.stats()
    assert "не подтверждённая человеком" in stats["note"]
    assert stats["candidates"] == 1
    store.close()


def test_export_dataset_writes_jsonl_and_summary_without_touching_protected_files(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.upsert_candidate("cycle_1", _candidate(cid="cand_in", included=True))
    store.upsert_candidate("cycle_1", _candidate(cid="cand_out", included=False))
    out_dir = tmp_path / "discovered"
    summary = store.export_dataset(out_dir, included_only=True)
    lines = (out_dir / "dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["candidate_id"] == "cand_in"
    assert summary["row_count"] == 1
    assert "не независимая экспертная разметка" in summary["provenance"]
    store.close()


def test_finish_cycle_records_error_status_without_raising(tmp_path):
    store = CorpusStore(tmp_path / "corpus.db")
    store.start_cycle("cycle_err", {"A": "topic a"}, ["A"])
    store.finish_cycle(
        "cycle_err", status="error",
        counts={"new_documents": 0, "new_candidates": 0, "included_candidates": 0, "excluded_candidates": 0},
        connectors_used=[], connectors_skipped=[], errors=[], error_message="boom",
    )
    cycles = store.recent_cycles()
    assert cycles[0]["status"] == "error"
    assert cycles[0]["error_message"] == "boom"
    store.close()
