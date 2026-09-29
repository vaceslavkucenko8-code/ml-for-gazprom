import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.schemas import Candidate, Evidence, SourceDocument, SourceType, stable_doc_id


def test_stable_doc_id_deterministic():
    assert stable_doc_id("https://example.com/a") == stable_doc_id("https://example.com/a")


def test_source_document_unknown_published_date_stays_none_not_fetched_at():
    doc = SourceDocument(
        doc_id="doc_1",
        url="https://example.com/a",
        normalized_url="https://example.com/a",
        text="текст без известной даты публикации",
        connector="test",
    )
    assert doc.published_at is None  # явно не подменяется fetched_at
    assert doc.fetched_at is not None


def test_candidate_has_no_final_classification_field():
    cand = Candidate(technology_name="X", short_description="Y", query="тема")
    dumped = cand.model_dump()
    assert "is_weak_signal" not in dumped
    assert "final_score" not in dumped
    assert "weak_signal_score" not in dumped


def test_evidence_serializes_with_quote_and_url():
    ev = Evidence(doc_id="doc_1", url="https://example.com/a", quote="точная цитата из текста")
    dumped = ev.model_dump()
    assert dumped["quote"] == "точная цитата из текста"
    assert dumped["url"] == "https://example.com/a"
