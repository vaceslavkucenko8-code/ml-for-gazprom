"""
src/retrieval/parse.py

Превращает "сырые" данные (HTML-страница ИЛИ уже структурированный SearchHit
от API-коннектора) в SourceDocument по общей схеме.

Извлекаются: оригинальное название, URL, дата публикации, язык, тип
источника, текст. Время загрузки (`fetched_at`) выставляется отдельно и
никогда не используется как замена неизвестной дате публикации —
`published_at=None` в этом случае остаётся None, а `published_at_is_estimated`
показывает, что дата была восстановлена эвристически (а не взята как есть).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import py3langid as langid
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

from .connectors.base import SearchHit
from .fetch import FetchResult
from .schemas import FetchStatus, SourceDocument, SourceType, now_utc, stable_doc_id

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# URL-нормализация (используется и здесь, и в deduplicate.py)
# --------------------------------------------------------------------------- #

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_referrer", "fbclid", "gclid", "yclid", "ysclid", "_openstat",
    "from", "ref", "referrer", "spm",
}


def normalize_url(url: str) -> str:
    """Приводит URL к каноническому виду для сравнения/дедупликации:
    убирает трекинговые параметры, фрагмент, завершающий слэш, приводит
    схему/хост к нижнему регистру, сортирует оставшиеся query-параметры.
    """
    from urllib.parse import parse_qsl, urlencode, urlunparse

    try:
        parts = urlparse(url.strip())
    except ValueError:
        return url.strip()

    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query_pairs.sort()
    query = urlencode(query_pairs)

    return urlunparse((scheme, netloc, path, "", query, ""))


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


# --------------------------------------------------------------------------- #
# Определение языка
# --------------------------------------------------------------------------- #


def detect_language(text: str, hint: Optional[str] = None) -> tuple[Optional[str], Optional[float]]:
    if hint:
        return hint[:2].lower(), 1.0
    text = (text or "").strip()
    if len(text) < 20:
        return None, None
    try:
        lang, score = langid.classify(text[:3000])
        # A log score is not a calibrated probability.
        return lang, None
    except Exception:  # noqa: BLE001
        return None, None


# --------------------------------------------------------------------------- #
# Определение типа источника по домену/URL (эвристика "по умолчанию")
# --------------------------------------------------------------------------- #

_PRIMARY_OFFICIAL_TLDS_HINTS = (".gov", ".gov.ru", ".europa.eu", ".un.org", ".who.int")
_ACADEMIC_HINTS = (
    "arxiv.org", "doi.org", "crossref.org", "ncbi.nlm.nih.gov", "springer.com",
    "nature.com", "sciencedirect.com", "ieee.org", "acm.org", "elibrary.ru",
    "cyberleninka.ru", "science.org", "frontiersin.org",
)
_PATENT_HINTS = ("patents.google.com", "fips.ru", "uspto.gov", "epo.org", "worldwide.espacenet.com")
_AGGREGATOR_HINTS = ("news.google.com", "news.yandex.ru", "yandex.ru/news", "news.rambler.ru", "dzen.ru", "smartnews")
_SOCIAL_HINTS = ("twitter.com", "x.com", "facebook.com", "t.me", "vk.com", "reddit.com", "linkedin.com")
_FORUM_HINTS = ("news.ycombinator.com", "habr.com/ru/companies",)  # habr — отдельно ниже


def guess_source_type(url: str, connector_default: Optional[str] = None) -> tuple[SourceType, float, str]:
    """Грубая эвристика по домену. Возвращает (тип, уверенность, обоснование).
    Это лишь стартовая гипотеза — deduplicate.py может её уточнить/подтвердить
    независимыми признаками; финальное объяснение всегда сохраняется в
    SourceDocument.source_type_rationale.
    """
    d = domain_of(url)
    # Editorial news section only; shops, forums and sponsored services differ.
    from urllib.parse import urlsplit
    if d in ('heise.de', 'www.heise.de') and urlsplit(url).path.startswith(('/news/', '/en/news/')):
        return SourceType.INDUSTRY_MEDIA, 0.8, 'Heise editorial technology news; publisher: https://www.heise.de/impressum.html; not a primary research paper'
    if d in ('news.mit.edu','research.birmingham.ac.uk'):
        return SourceType.UNIVERSITY, 0.9, f'Institutional research publication ({d}); not independent peer review'
    def matches(hints):
        return any(d == h.lstrip('.') or d.endswith('.' + h.lstrip('.')) for h in hints)

    if matches(_PRIMARY_OFFICIAL_TLDS_HINTS):
        return SourceType.PRIMARY_OFFICIAL, 0.85, f"домен верхнего уровня/паттерн указывает на госорган/регулятора ({d})"

    if matches(_PATENT_HINTS):
        return SourceType.PATENT, 0.9, f"домен патентной базы ({d})"

    if matches(_ACADEMIC_HINTS):
        return SourceType.ACADEMIC, 0.9, f"домен научного издательства/архива препринтов ({d})"

    if matches(_AGGREGATOR_HINTS):
        return SourceType.AGGREGATOR, 0.75, f"домен новостного агрегатора ({d})"

    if matches(_SOCIAL_HINTS):
        return SourceType.SOCIAL_MEDIA, 0.9, f"домен социальной сети/мессенджера ({d})"

    if d == "github.com":
        return SourceType.CODE_REPOSITORY, 0.95, "репозиторий на GitHub — сигнал разработческой активности"

    if d == "news.ycombinator.com":
        return SourceType.FORUM, 0.9, "обсуждение на Hacker News — не первоисточник, а форумный сигнал"

    if connector_default == "developer_publication":
        return SourceType.DEVELOPER_PUBLICATION, 0.6, "получено коннектором официальных публикаций разработчика"

    # по умолчанию — общее медиа с низкой уверенностью, требует уточнения
    return SourceType.GENERAL_MEDIA, 0.35, f"тип определён по умолчанию для домена {d}; требуется ручная/доп. проверка"


# --------------------------------------------------------------------------- #
# Извлечение основного текста и метаданных из HTML
# --------------------------------------------------------------------------- #

_META_DATE_NAMES = (
    "article:published_time", "og:article:published_time", "publish-date",
    "publishdate", "date", "dc.date", "dc.date.issued",
    "parsely-pub-date", "sailthru.date", "citation_publication_date", "citation_date",
    "dcterms.date", "dcterms.issued", "citation_online_date",
)


def _extract_html_meta(soup: BeautifulSoup) -> dict:
    meta = {}
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or "").lower()
        content = tag.get("content")
        if key and content:
            meta[key] = content
    return meta


def _as_utc_aware(dt: datetime) -> datetime:
    # Многие сайты пишут дату без указания часового пояса (например,
    # "2024-01-15T10:00:00"). Остальной конвейер (fetched_at, коннекторы)
    # всегда работает с tz-aware UTC-датами — наивная дата иначе валит
    # сравнение дат при дедупликации (TypeError: naive vs aware). Полагаем
    # такую дату уже UTC, а не гадаем настоящий часовой пояс издателя.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_published_at(soup: BeautifulSoup, meta: dict) -> tuple[Optional[datetime], bool]:
    def complete_date(value):
        # Missing components must not silently come from today's date.
        first = dateutil_parser.parse(value, default=datetime(1901, 1, 1))
        second = dateutil_parser.parse(value, default=datetime(1902, 2, 2))
        if first.date() != second.date():
            raise ValueError('Incomplete publication date')
        return _as_utc_aware(first)
    for name in _META_DATE_NAMES:
        if name in meta:
            try:
                return complete_date(meta[name]), False
            except (ValueError, OverflowError):
                continue

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            return complete_date(time_tag["datetime"]), False
        except (ValueError, OverflowError):
            pass

    # эвристика последней надежды: дата в самом URL (/2026/08/29/...)
    return None, False


def _extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    body = soup.select_one('.news-article--content--body--inner, [itemprop="articleBody"]')
    candidates = [body] if body else soup.find_all('article') or soup.find_all('main') or []
    # Never concatenate a parent container and its nested article a second time.
    candidates = [c for c in candidates if not any(p in candidates for p in c.parents)]
    if not candidates:
        # эвристика по плотности текста: контейнер с наибольшим количеством
        # текста в <p> считается основным контентом
        best, best_len = None, 0
        for div in soup.find_all(["div", "section"]):
            length = len(div.get_text(strip=True))
            if length > best_len:
                best, best_len = div, length
        candidates = [best] if best else []

    if candidates:
        text = "\n".join(c.get_text(separator="\n", strip=True) for c in candidates if c)
    else:
        text = soup.get_text(separator="\n", strip=True)

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 1]
    return "\n".join(lines)


def parse_html_document(
    fetch_result: FetchResult,
    hit: SearchHit,
) -> SourceDocument:
    """Разбор HTML-страницы, полученной через fetch.py."""
    url = fetch_result.final_url or fetch_result.url
    normalized = normalize_url(url)
    doc_id = stable_doc_id(normalized)

    if fetch_result.status != FetchStatus.OK or not fetch_result.content:
        return SourceDocument(
            doc_id=doc_id,
            url=url,
            normalized_url=normalized,
            title=hit.title,
            connector=hit.connector,
            query=hit.query,
            source_type=SourceType.UNKNOWN,
            fetch_status=fetch_result.status,
            fetch_error=fetch_result.error,
            published_at=hit.published_at,
            fetched_at=now_utc(),
        )

    soup = BeautifulSoup(fetch_result.content, "lxml")
    meta = _extract_html_meta(soup)
    from ..source_provenance import work_id
    from urllib.parse import urljoin
    research_links = sorted({urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                             if work_id(urljoin(url, a['href']))})

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    title = meta.get("og:title") or title or hit.title

    # Main-text extraction removes headers, including publication <time> tags.
    published_at, is_estimated = _extract_published_at(soup, meta)
    text = _extract_main_text(soup)
    text_scope = 'full_page' if text else 'search_snippet'
    if not text and hit.snippet:
        text = hit.snippet

    if published_at is None and hit.published_at is not None:
        published_at, is_estimated = hit.published_at, False

    lang_hint = meta.get("og:locale", "")[:2] or hit.language_hint
    language, lang_conf = detect_language(text or title or "", hint=lang_hint)

    source_type, type_conf, rationale = guess_source_type(url, hit.connector)

    content_hash = hashlib.sha256((text or title or url).encode("utf-8", errors="ignore")).hexdigest()

    return SourceDocument(
        doc_id=doc_id,
        url=url,
        normalized_url=normalized,
        title=title,
        text=text or "",
        language=language,
        language_confidence=lang_conf,
        source_type=source_type,
        source_type_confidence=type_conf,
        source_type_rationale=rationale,
        source_name=meta.get("og:site_name") or domain_of(url),
        domain=domain_of(url),
        published_at=published_at,
        published_at_is_estimated=is_estimated,
        fetched_at=now_utc(),
        connector=hit.connector,
        query=hit.query,
        fetch_status=FetchStatus.OK,
        content_hash=content_hash,
        raw_metadata={"meta": {k: v for k, v in list(meta.items())[:30]}, **hit.raw,
                      'text_scope': text_scope, 'research_links': research_links},
    )


def parse_api_hit(hit: SearchHit, default_source_type: SourceType) -> SourceDocument:
    """Разбор находки, которая пришла уже со структурированным текстом от
    самого API (arXiv, Crossref, GitHub, ...) — без дополнительного HTTP
    запроса на HTML-страницу.
    """
    normalized = normalize_url(hit.url)
    doc_id = stable_doc_id(normalized)
    original_text = hit.full_text or hit.snippet or ""
    text = BeautifulSoup(original_text, 'html.parser').get_text(' ', strip=True) if '<' in original_text else original_text

    language, lang_conf = detect_language(text or hit.title or "", hint=hit.language_hint)
    domain_guess_type, type_conf, rationale = guess_source_type(hit.url, hit.connector)
    # для API-коннекторов домен часто не даёт полезной эвристики (например,
    # export.arxiv.org) — предпочитаем default_source_type самого коннектора,
    # если доменная эвристика вернула низкую уверенность
    if type_conf < 0.6:
        source_type = default_source_type
        rationale = f"тип определён коннектором «{hit.connector}» по умолчанию: {default_source_type.value}"
        type_conf = 0.7
    else:
        source_type = domain_guess_type

    content_hash = hashlib.sha256((text or hit.title or hit.url).encode("utf-8", errors="ignore")).hexdigest()

    return SourceDocument(
        doc_id=doc_id,
        url=hit.url,
        normalized_url=normalized,
        title=hit.title,
        text=text,
        language=language,
        language_confidence=lang_conf,
        source_type=source_type,
        source_type_confidence=type_conf,
        source_type_rationale=rationale,
        source_name=domain_of(hit.url),
        domain=domain_of(hit.url),
        published_at=hit.published_at,
        published_at_is_estimated=bool(hit.raw.get('date_estimated')),
        fetched_at=now_utc(),
        connector=hit.connector,
        query=hit.query,
        fetch_status=FetchStatus.OK,
        content_hash=content_hash,
        raw_metadata={**hit.raw, 'text_scope': hit.raw.get('text_scope') or ('abstract_or_description' if hit.full_text else 'search_snippet'),
                      'api_original_text': original_text},
    )
