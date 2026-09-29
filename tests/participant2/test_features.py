import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.candidates import extract_candidates
from src.extraction.features import candidate_features, explain_features
from src.retrieval.deduplicate import deduplicate_and_score
from src.retrieval.parse import normalize_url
from src.retrieval.schemas import SourceDocument, SourceType, now_utc, stable_doc_id


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


def test_features_are_plain_numbers_not_a_black_box_score():
    text = (
        "Компания QNav Systems провела пилотное внедрение квантового сенсора без GPS. "
        "В отличие от GPS, сенсор не требует спутникового сигнала. "
        "Компания планирует выпустить коммерческую версию к 2028 году."
    )
    doc1 = make_doc("https://a.com/1", text, title="QNav Systems демонстрирует пилот", source_type=SourceType.ACADEMIC,
                     published_at=datetime(2026, 1, 1))
    doc2 = make_doc("https://b.com/1", text, title="QNav Systems демонстрирует пилот", source_type=SourceType.INDUSTRY_MEDIA,
                     published_at=datetime(2026, 3, 1))
    docs = deduplicate_and_score([doc1, doc2])
    candidates = extract_candidates("квантовая навигация", docs)
    assert len(candidates) == 1
    cand = candidates[0]

    features = candidate_features(cand, docs)

    # признаки — обычные python-типы (int/float/bool/None), никакой
    # "чёрный ящик" score от сторонней библиотеки здесь не участвует
    for key, value in features.items():
        assert value is None or isinstance(value, (int, float, bool))

    assert features["has_academic_or_patent_evidence"] is True
    assert isinstance(features["future_looking_ratio"], float)
    assert 0.0 <= features["future_looking_ratio"] <= 1.0

    explanations = explain_features(features)
    assert len(explanations) == len(features)
    assert all(isinstance(line, str) and "=" in line for line in explanations)


def test_lowest_trust_only_candidate_has_low_signal_features():
    doc = make_doc(
        "https://blog.example/1",
        "Кто-то написал в личном блоге про новую технологию без единого подтверждения.",
        source_type=SourceType.PERSONAL_BLOG,
    )
    docs = deduplicate_and_score([doc])
    candidates = extract_candidates("тема", docs)
    features = candidate_features(candidates[0], docs)
    assert features["lowest_trust_only"] is True
    assert features["high_trust_source_count"] == 0
    assert features["independent_confirmations"] == 0
