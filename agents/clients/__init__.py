"""
Data Source Clients for FinAgent
Provides clean interfaces to external data sources
"""

from .yahoo_client import YahooFinanceClient, get_yahoo_client
from .news_api import NewsClient, get_news_client
from .backend_client import BackendClient, get_backend_client
from .tradingview_client import TradingViewClient, get_tradingview_client

__all__ = [
    # Classes
    'YahooFinanceClient',
    'NewsClient',
    'BackendClient',
    'TradingViewClient',
    
    # Singleton getters
    'get_yahoo_client',
    'get_news_client',
    'get_backend_client',
    'get_tradingview_client',
]