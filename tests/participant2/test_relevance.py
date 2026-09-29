import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.candidates import extract_candidates
from src.extraction.relevance import (
    RelevanceVerdict,
    assess_candidates,
    assess_relevance,
    explain_relevance,
)
from src.retrieval.deduplicate import deduplicate_and_score
from src.retrieval.parse import normalize_url
from src.retrieval.schemas import SourceDocument, SourceType, now_utc, stable_doc_id

NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)


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


def test_score_is_within_0_and_1_and_verdict_is_enum():
    doc = make_doc(
        "https://blog.example/1",
        "Кто-то написал в личном блоге про новую технологию без единого подтверждения.",
        source_type=SourceType.PERSONAL_BLOG,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема", docs)
    assessment = assess_relevance(candidates[0], docs, now=NOW)
    assert 0.0 <= assessment.score <= 1.0
    assert isinstance(assessment.verdict, RelevanceVerdict)


def test_lowest_trust_only_without_confirmation_is_noise_or_insufficient():
    doc = make_doc(
        "https://blog.example/1",
        "Кто-то написал в личном блоге про новую технологию без единого подтверждения.",
        source_type=SourceType.PERSONAL_BLOG,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема", docs)
    assessment = assess_relevance(candidates[0], docs, now=NOW)
    assert assessment.verdict in (RelevanceVerdict.LIKELY_NOISE, RelevanceVerdict.INSUFFICIENT_EVIDENCE)
    assert assessment.score < 0.65


def test_strong_multi_source_recent_candidate_is_likely_signal_and_actual():
    text_a = (
        "QNav Systems провела пилотное внедрение квантового сенсора без GPS. "
        "В отличие от GPS, сенсор не требует спутникового сигнала."
    )
    text_b = (
        "Независимое издание подтверждает: QNav Systems провела пилотное внедрение "
        "квантового сенсора без GPS, испытания прошли успешно."
    )
    recent_date = NOW - timedelta(days=30)
    doc1 = make_doc(
        "https://academic.example/paper", text_a, title="QNav Systems демонстрирует пилот",
        source_type=SourceType.ACADEMIC, published_at=recent_date,
    )
    doc2 = make_doc(
        "https://othernews.example/article", text_b, title="QNav Systems демонстрирует пилот",
        source_type=SourceType.INDUSTRY_MEDIA, published_at=recent_date + timedelta(days=1),
    )
    docs = deduplicate_and_score([doc1, doc2])
    candidates = extract_candidates("квантовая навигация", docs)
    assert len(candidates) == 1
    assessment = assess_relevance(candidates[0], docs, now=NOW)

    assert assessment.verdict == RelevanceVerdict.LIKELY_EMERGING_SIGNAL
    assert assessment.is_actual is True
    assert any(c.weight > 0 for c in assessment.contributions)


def test_old_evidence_without_recent_confirmation_is_not_actual():
    old_date = NOW - timedelta(days=800)
    doc = make_doc(
        "https://academic.example/old-paper",
        "Компания провела пилотное внедрение технологии давно.",
        title="Старый пилот технологии",
        source_type=SourceType.ACADEMIC,
        published_at=old_date,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("старая тема", docs)
    assessment = assess_relevance(candidates[0], docs, now=NOW)
    assert assessment.is_actual is False
    assert "устар" in assessment.actuality_reason or "угас" in assessment.actuality_reason


def test_missing_dates_gives_unknown_actuality_not_a_guess():
    doc = make_doc(
        "https://academic.example/no-date",
        "Компания провела пилотное внедрение технологии.",
        title="Пилот без даты",
        source_type=SourceType.ACADEMIC,
        published_at=None,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема без даты", docs)
    assessment = assess_relevance(candidates[0], docs, now=NOW)
    assert assessment.is_actual is None


def test_explain_relevance_returns_readable_lines_for_every_contribution():
    doc = make_doc(
        "https://blog.example/1",
        "Личный блог про технологию.",
        source_type=SourceType.PERSONAL_BLOG,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема", docs)
    assessment = assess_relevance(candidates[0], docs, now=NOW)
    lines = explain_relevance(assessment)
    assert len(lines) == len(assessment.contributions) + 1
    assert all(isinstance(line, str) for line in lines)


def test_assess_candidates_batches_over_all_candidates():
    doc_a = make_doc(
        "https://a.example/1", "Пилотное внедрение нейроморфного чипа.",
        title="Neuromorphic Chip Labs представила пилот", source_type=SourceType.ACADEMIC,
    )
    doc_b = make_doc(
        "https://b.example/1", "Патентная заявка на постквантовый криптографический протокол.",
        title="Postquantum Crypto Systems подала патентную заявку", source_type=SourceType.PATENT,
    )
    docs = deduplicate_and_score([doc_a, doc_b])
    candidates = extract_candidates("две разные темы", docs)
    assert len(candidates) == 2
    assessments = assess_candidates(candidates, docs, now=NOW)
    assert len(assessments) == 2
    assert {a.candidate_id for a in assessments} == {c.candidate_id for c in candidates}


def test_relevance_assessment_does_not_mutate_or_extend_candidate_schema():
    doc = make_doc("https://a.example/1", "Технология пилот.", source_type=SourceType.ACADEMIC)
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема", docs)
    dumped_before = set(candidates[0].model_dump().keys())
    assess_relevance(candidates[0], docs, now=NOW)
    dumped_after = set(candidates[0].model_dump().keys())
    assert dumped_before == dumped_after
    assert "is_weak_signal" not in dumped_after
    assert "final_score" not in dumped_after
