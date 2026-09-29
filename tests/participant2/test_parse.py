import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.connectors.base import SearchHit
from src.retrieval.fetch import FetchResult
from src.retrieval.parse import (
    detect_language,
    domain_of,
    guess_source_type,
    normalize_url,
    parse_html_document,
)
from src.retrieval.schemas import FetchStatus, SourceType


def test_normalize_url_basic():
    assert normalize_url("https://Example.com/Path/") == "https://example.com/Path"
    assert normalize_url("http://www.example.com/x?b=2&a=1") == "http://example.com/x?a=1&b=2"


def test_domain_of():
    assert domain_of("https://www.habr.com/ru/articles/1/") == "habr.com"


def test_detect_language_uses_hint_first():
    lang, conf = detect_language("irrelevant text", hint="ru")
    assert lang == "ru"
    assert conf == 1.0


def test_detect_language_detects_russian_text():
    lang, conf = detect_language("Это пример достаточно длинного русского текста для определения языка." * 3)
    assert lang == "ru"


def test_guess_source_type_academic_domain():
    st, conf, rationale = guess_source_type("https://arxiv.org/abs/1234.5678")
    assert st == SourceType.ACADEMIC
    assert conf > 0.5


def test_guess_source_type_patent_domain():
    st, conf, rationale = guess_source_type("https://patents.google.com/patent/US123")
    assert st == SourceType.PATENT


def test_guess_source_type_social_media():
    st, conf, rationale = guess_source_type("https://x.com/someone/status/1")
    assert st == SourceType.SOCIAL_MEDIA


def test_parse_html_document_failed_fetch_preserves_query_and_marks_status():
    hit = SearchHit(url="https://unreachable.example/page", title="Заголовок", connector="gdelt", query="тема")
    failed = FetchResult(url=hit.url, status=FetchStatus.TIMEOUT, error="таймаут соединения")
    doc = parse_html_document(failed, hit)
    assert doc.fetch_status == FetchStatus.TIMEOUT
    assert doc.fetch_error == "таймаут соединения"
    assert doc.query == "тема"
    assert doc.title == "Заголовок"


def test_parse_html_document_extracts_title_and_text():
    html = """
    <html><head>
      <title>Тестовая статья</title>
      <meta property="og:site_name" content="Test Media">
      <meta property="article:published_time" content="2026-05-01T10:00:00Z">
    </head>
    <body>
      <nav>меню сайта, не относится к статье</nav>
      <article>
        <p>Компания представила новую технологию квантовой навигации без GPS.</p>
        <p>Технология находится на стадии пилотного внедрения в нескольких портах.</p>
      </article>
      <footer>подвал сайта</footer>
    </body></html>
    """
    hit = SearchHit(url="https://example.com/news/1", title=None, connector="gdelt", query="квант навигация")
    fetch_result = FetchResult(
        url=hit.url, status=FetchStatus.OK, content=html, content_type="text/html", final_url=hit.url,
    )
    doc = parse_html_document(fetch_result, hit)
    assert doc.title == "Тестовая статья"
    assert "квантовой навигации" in doc.text
    assert "меню сайта" not in doc.text
    assert "подвал сайта" not in doc.text
    assert doc.published_at is not None
    assert doc.published_at.tzinfo is not None
    assert doc.source_name == "Test Media"


def test_parse_html_document_naive_meta_date_becomes_utc_aware():
    # Многие сайты пишут дату без указания часового пояса
    # (например, dc.date.issued="2026-05-01T10:00:00", без "Z"/смещения).
    # Остальной конвейер (fetched_at, коннекторы arXiv/GitHub/HackerNews)
    # всегда tz-aware UTC — naive дата иначе валит сравнение дат в
    # deduplicate.py (TypeError: naive vs aware).
    html = """
    <html><head>
      <title>Статья без явного часового пояса</title>
      <meta name="dc.date.issued" content="2026-05-01T10:00:00">
    </head>
    <body><article><p>Текст статьи о квантовой навигации без GPS.</p></article></body></html>
    """
    hit = SearchHit(url="https://example.com/news/2", title=None, connector="gdelt", query="квант навигация")
    fetch_result = FetchResult(
        url=hit.url, status=FetchStatus.OK, content=html, content_type="text/html", final_url=hit.url,
    )
    doc = parse_html_document(fetch_result, hit)
    assert doc.published_at is not None
    assert doc.published_at.tzinfo is not None
