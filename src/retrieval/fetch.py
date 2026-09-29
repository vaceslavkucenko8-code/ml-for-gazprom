"""
src/retrieval/fetch.py

Единая точка HTTP-доступа для всего модуля поиска/загрузки.

Требования ТЗ (п. 2.2), которые здесь реализованы:
  * тайм-ауты (отдельно connect / read);
  * ограниченное число повторов с экспоненциальным backoff, только для
    "временных" ошибок (5xx, сетевые сбои, таймауты) — не для 4xx;
  * понятный, типизированный результат при недоступной странице —
    FetchResult с fetch_status/fetch_error вместо необработанного исключения,
    чтобы ошибка одной страницы не могла "уронить" весь сбор;
  * вежливость к источникам: свой User-Agent, минимальный интервал между
    запросами к одному домену (мягкий rate-limit на стороне клиента).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import UnicodeDammit
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .schemas import FetchStatus

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "SignalScout/1.0 (technology research)"

DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MIN_DELAY_PER_DOMAIN = 1.0  # секунд между запросами к одному домену


def decode_response(content: bytes, content_type: str) -> str:
    """Honor explicit encodings; don't use requests' HTML Latin-1 default."""
    if 'application/json' in content_type.lower():
        return content.decode('utf-8-sig')
    declared = re.search(r'charset\s*=\s*[\"\']?([^\s;\"\']+)', content_type, re.I)
    if declared:
        try:
            return content.decode(declared.group(1))
        except (LookupError, UnicodeDecodeError):
            pass
    decoded = UnicodeDammit(content, is_html='html' in content_type.lower()).unicode_markup
    if decoded is None:
        raise UnicodeError('Unknown response encoding')
    return decoded


@dataclasses.dataclass
class FetchResult:
    url: str
    status: FetchStatus
    status_code: Optional[int] = None
    content: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    final_url: Optional[str] = None  # после редиректов


class _DomainThrottle:
    """Простейший клиентский rate-limit: не бьём один домен чаще, чем раз в
    `min_delay` секунд. Потокобезопасно (на случай параллельного сбора).
    """

    def __init__(self, min_delay: float = DEFAULT_MIN_DELAY_PER_DOMAIN):
        self.min_delay = min_delay
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        with self._lock:
            last = self._last_call.get(domain, 0.0)
            now = time.monotonic()
            delta = now - last
            sleep_for = self.min_delay - delta
            self._last_call[domain] = now + max(sleep_for, 0.0)
        if sleep_for > 0:
            time.sleep(sleep_for)


class Fetcher:
    """HTTP-клиент с ретраями, таймаутами и понятной обработкой ошибок."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_delay_per_domain: float = DEFAULT_MIN_DELAY_PER_DOMAIN,
        extra_headers: Optional[dict] = None,
    ):
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.throttle = _DomainThrottle(min_delay_per_domain)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml,"
                "application/json,application/atom+xml;q=0.9,*/*;q=0.7",
                "Accept-Language": "ru,en;q=0.8",
            }
        )
        if extra_headers:
            self.session.headers.update(extra_headers)

        retry = Retry(
            total=max_retries,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> FetchResult:
        domain = urlparse(url).netloc
        self.throttle.wait(domain)

        t0 = time.monotonic()
        try:
            resp = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=(self.connect_timeout, self.read_timeout),
                allow_redirects=True,
            )
        except requests.exceptions.Timeout as exc:
            return FetchResult(
                url=url, status=FetchStatus.TIMEOUT, error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
        except requests.exceptions.ConnectionError as exc:
            return FetchResult(
                url=url, status=FetchStatus.CONNECTION_ERROR, error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )
        except requests.exceptions.RequestException as exc:
            return FetchResult(
                url=url, status=FetchStatus.CONNECTION_ERROR, error=str(exc),
                elapsed_seconds=time.monotonic() - t0,
            )

        elapsed = time.monotonic() - t0

        if resp.status_code in (401, 403, 451):
            return FetchResult(
                url=url, status=FetchStatus.BLOCKED, status_code=resp.status_code,
                error=f"HTTP {resp.status_code}: доступ заблокирован (авторизация/paywall/гео-ограничение)",
                elapsed_seconds=elapsed, final_url=resp.url,
            )
        if resp.status_code >= 400:
            return FetchResult(
                url=url, status=FetchStatus.HTTP_ERROR, status_code=resp.status_code,
                error=f"HTTP {resp.status_code}", elapsed_seconds=elapsed, final_url=resp.url,
            )

        content_type = resp.headers.get("Content-Type", "")
        try:
            text = decode_response(resp.content, content_type)
        except Exception as exc:  # noqa: BLE001 - любая ошибка декодирования не должна валить сбор
            return FetchResult(
                url=url, status=FetchStatus.PARSE_ERROR, status_code=resp.status_code,
                error=f"Не удалось декодировать ответ: {exc}", elapsed_seconds=elapsed,
                final_url=resp.url,
            )

        return FetchResult(
            url=url, status=FetchStatus.OK, status_code=resp.status_code,
            content=text, content_type=content_type, elapsed_seconds=elapsed,
            final_url=resp.url,
        )


_default_fetcher: Optional[Fetcher] = None


def get_default_fetcher() -> Fetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = Fetcher()
    return _default_fetcher
