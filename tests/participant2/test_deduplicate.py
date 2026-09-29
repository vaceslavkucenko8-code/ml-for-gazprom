import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.deduplicate import (
    annotate_duplicates,
    cluster_by_event,
    count_independent_confirmations,
    deduplicate_and_score,
    deduplicate_exact,
)
from src.retrieval.parse import normalize_url
from src.retrieval.schemas import SourceDocument, SourceType, TrustLevel, now_utc, stable_doc_id


def make_doc(url, text, title=None, source_type=SourceType.GENERAL_MEDIA, published_at=None, **kw):
    normalized = normalize_url(url)
    return SourceDocument(
        doc_id=stable_doc_id(normalized),
        url=url,
        normalized_url=normalized,
        title=title or text[:40],
        text=text,
        source_type=source_type,
        source_type_confidence=0.8,
        published_at=published_at,
        fetched_at=now_utc(),
        connector="test",
        **kw,
    )


def test_normalize_url_strips_tracking_params():
    a = normalize_url("HTTPS://WWW.Example.com/Article/?utm_source=tw&id=5")
    b = normalize_url("https://example.com/Article?id=5")
    assert a == b


def test_exact_duplicate_by_url():
    docs = [
        make_doc("https://example.com/a?utm_source=x", "текст статьи один"),
        make_doc("https://example.com/a", "текст статьи один, но чуть длиннее версия"),
    ]
    result = deduplicate_exact(docs)
    assert len(result) == 1


def test_near_duplicate_marks_duplicate_of():
    text = "Компания X привлекла 10 миллионов долларов на разработку квантовых сенсоров для навигации без GPS. " * 3
    docs = [
        make_doc("https://a.com/1", text, source_type=SourceType.INDUSTRY_MEDIA),
        make_doc("https://mirror-a.com/1", text, source_type=SourceType.AGGREGATOR),
    ]
    annotate_duplicates(docs)
    dup_flags = [d.duplicate_of for d in docs]
    assert any(dup_flags)  # хотя бы один помечен как дубль
    # канонический должен быть industry_media (выше приоритет), а не aggregator
    canonical = [d for d in docs if d.duplicate_of is None][0]
    assert canonical.source_type == SourceType.INDUSTRY_MEDIA


def test_report_news_and_report_pdf_is_one_confirmation():
    """Ключевое требование ТЗ: новость об отчёте и сам отчёт — не два
    независимых подтверждения, если это один и тот же инфоповод."""
    now = datetime(2026, 9, 1)
    news = make_doc(
        "https://news-site.com/report-launch",
        "Аналитическая компания опубликовала отчёт о рынке квантовой навигации. "
        "В отчёте говорится о росте инвестиций в квантовые сенсоры на 40% за год.",
        source_type=SourceType.GENERAL_MEDIA,
        published_at=now,
    )
    report_pdf = make_doc(
        "https://analyst.com/reports/quantum-navigation-2026.pdf",
        "В отчёте говорится о росте инвестиций в квантовые сенсоры на 40% за год. "
        "Аналитическая компания опубликовала отчёт о рынке квантовой навигации.",
        source_type=SourceType.ANALYST_REPORT,
        published_at=now + timedelta(days=1),
    )
    docs = [news, report_pdf]
    cluster_by_event(docs)
    assert news.event_cluster_id == report_pdf.event_cluster_id

    result = count_independent_confirmations(docs)
    assert result.independent_confirmations == 1


def test_independent_confirmation_across_distinct_events():
    docs = [
        make_doc("https://a.com/1", "Первая новость про совсем другое событие A " * 5,
                 source_type=SourceType.INDUSTRY_MEDIA, published_at=datetime(2026, 1, 1)),
        make_doc("https://b.com/1", "Совершенно другое событие B, никак не пересекается по содержанию " * 5,
                 source_type=SourceType.INDUSTRY_MEDIA, published_at=datetime(2026, 6, 1)),
    ]
    result = count_independent_confirmations(docs)
    assert result.independent_confirmations == 2


def test_low_trust_single_domain_not_independent_confirmation():
    docs = [
        make_doc("https://blog.com/1", "Личный блог про технологию, без подтверждений " * 5,
                 source_type=SourceType.PERSONAL_BLOG),
    ]
    result = count_independent_confirmations(docs)
    assert result.independent_confirmations == 0


def test_trust_level_high_for_academic():
    doc = make_doc("https://arxiv.org/abs/1234", "препринт статья про метод " * 5, source_type=SourceType.ACADEMIC)
    scored = deduplicate_and_score([doc])
    assert scored[0].trust_level == TrustLevel.HIGH


def test_trust_level_low_for_lone_social_media():
    doc = make_doc("https://x.com/user/status/1", "твит про технологию без подтверждений " * 3,
                    source_type=SourceType.SOCIAL_MEDIA)
    scored = deduplicate_and_score([doc])
    assert scored[0].trust_level == TrustLevel.LOW
