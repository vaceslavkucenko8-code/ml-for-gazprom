import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.candidates import (
    extract_candidates,
    extract_companies,
    is_future_looking,
    split_sentences,
)
from src.retrieval.deduplicate import deduplicate_and_score
from src.retrieval.parse import normalize_url
from src.retrieval.schemas import ClaimType, SourceDocument, SourceType, now_utc, stable_doc_id


def make_doc(url, text, title=None, source_type=SourceType.INDUSTRY_MEDIA, published_at=None):
    normalized = normalize_url(url)
    return SourceDocument(
        doc_id=stable_doc_id(normalized),
        url=url,
        normalized_url=normalized,
        title=title or text[:50],
        text=text,
        source_type=source_type,
        source_type_confidence=0.8,
        published_at=published_at,
        fetched_at=now_utc(),
        connector="test",
        query="тестовая тема",
    )


def test_split_sentences_ru_en_mixed():
    text = "Первое предложение про технологию. Second sentence about the same topic. Третье."
    sentences = split_sentences(text)
    assert len(sentences) >= 2


def test_is_future_looking_detects_plans():
    assert is_future_looking("Компания планирует запустить пилот в 2027 году.")
    assert is_future_looking("The company plans to launch a pilot by 2027.")
    assert not is_future_looking("Компания запустила пилот в 2026 году.")


def test_extract_companies_finds_org_suffix():
    companies = extract_companies("Dirac Labs raised $1.8M for quantum sensors.", "Dirac Labs Raises $1.8M")
    assert "Dirac Labs" in companies


def test_extract_candidates_end_to_end_separates_fact_from_plan():
    text = (
        "Компания QNav Systems провела пилотное внедрение квантового сенсора без GPS на судне. "
        "В отличие от GPS, сенсор не требует спутникового сигнала. "
        "Компания планирует выпустить коммерческую версию к 2028 году."
    )
    doc = make_doc("https://example.com/qnav", text, title="QNav Systems демонстрирует пилот квантовой навигации")
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("квантовая навигация", docs)

    assert len(candidates) == 1
    cand = candidates[0]
    assert "QNav" in cand.technology_name

    facts = [e for e in cand.evidence if e.claim_type == ClaimType.STAGE_OF_DEVELOPMENT]
    assert any(not e.is_future_looking for e in facts)  # пилот — уже произошедшее событие
    assert any(e.is_future_looking for e in facts)  # план на 2028 год — будущее

    assert len(cand.advantages) >= 1

    # у Candidate НЕТ финального класса "слабый сигнал" — это не наша задача
    assert not hasattr(cand, "is_weak_signal")
    assert not hasattr(cand, "final_score")


def test_extract_candidates_clusters_separate_topics_into_separate_candidates():
    doc1 = make_doc(
        "https://a.com/1",
        "Dirac Labs привлекла посевное финансирование на разработку квантовых сенсоров.",
        title="Dirac Labs привлекла посевное финансирование",
    )
    doc2 = make_doc(
        "https://b.com/1",
        "Совершенно другая компания CoolBio разработала биопринтер нового поколения для тканей.",
        title="CoolBio представила биопринтер нового поколения",
    )
    docs = deduplicate_and_score([doc1, doc2])
    candidates = extract_candidates("разное", docs)
    assert len(candidates) == 2
