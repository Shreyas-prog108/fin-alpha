"""
Data Source Clients for FinAgent
Provides clean interfaces to external data sources
"""

from .backend_client import BackendClient, get_backend_client
from .alphavantage_client import AlphaVantageClient, get_alphavantage_client
from .mint_client import MintClient, get_mint_client
from .perplexity_client import PerplexityClient, get_perplexity_client
from .news_api import NewsClient, get_news_client
from .google_search_client import GoogleSearchClient, get_google_search_client

__all__ = [
    # Classes
    'BackendClient',
    'AlphaVantageClient',
    'MintClient',
    'PerplexityClient',
    'NewsClient',
    'GoogleSearchClient',

    # Singleton getters
    'get_backend_client',
    'get_alphavantage_client',
    'get_mint_client',
    'get_perplexity_client',
    'get_news_client',
    'get_google_search_client',
]
