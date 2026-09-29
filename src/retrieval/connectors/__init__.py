from .arxiv import ArxivConnector
from .base import ConnectorError, SearchConnector, SearchHit
from .commercial import TavilyConnector
from .crossref import CrossrefConnector
from .gdelt import GdeltConnector
from .github import GitHubConnector
from .hackernews import HackerNewsConnector
from .mit_news import MitNewsConnector

#: коннекторы, включённые по умолчанию (все бесплатные, без ключа,
#: с официально задокументированными условиями доступа)
DEFAULT_CONNECTORS = [
    MitNewsConnector,
    ArxivConnector,
    CrossrefConnector,
    GdeltConnector,
    HackerNewsConnector,
    GitHubConnector,
]

#: опциональные коннекторы, включаются только при наличии ключа в окружении
OPTIONAL_CONNECTORS = [
    TavilyConnector,
]

__all__ = [
    "SearchConnector",
    "SearchHit",
    "ConnectorError",
    "ArxivConnector",
    "CrossrefConnector",
    "GdeltConnector",
    "HackerNewsConnector",
    "GitHubConnector",
    "TavilyConnector",
    "DEFAULT_CONNECTORS",
    "OPTIONAL_CONNECTORS",
]
