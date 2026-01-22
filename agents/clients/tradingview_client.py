"""
TradingView Client
Fetches real-time market data, technical indicators, and company information
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime


class TradingViewClient:
    """
    TradingView data client
    Provides real-time quotes, technical analysis, and market data
    """
    
    def __init__(self):
        self.search_url = os.getenv(
            "TRADINGVIEW_API_URL",
            "https://symbol-search.tradingview.com/symbol_search/"
        )
        self.quote_url = "https://scanner.tradingview.com/symbol"
        self.cache = {}
        self.cache_ttl = 300
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Set cache with current timestamp"""
        self.cache[key] = (data, datetime.now().timestamp())
    
    def search_symbol(self, query: str) -> Optional[Dict]:
        """
        Search for a ticker symbol
        
        Args:
            query: Company name or ticker symbol
        
        Returns:
            Dictionary with symbol info or None if not found
        """
        try:
            params = {
                'text': query,
                'type': 'stock',
                'exchange': '',
                'lang': 'en'
            }
            
            response = requests.get(self.search_url, params=params, timeout=5)
            if response.status_code != 200:
                raise Exception(f"Search API returned status {response.status_code}")
            try:
                data = response.json()
            except ValueError as e:
                raise Exception(f"Invalid JSON response: {str(e)}")
            
            if data and len(data) > 0:
                best_match = data[0]
                return {
                    'symbol': best_match.get('symbol', ''),
                    'description': best_match.get('description', ''),
                    'exchange': best_match.get('exchange', ''),
                    'type': best_match.get('type', ''),
                    'ticker': best_match.get('symbol', '').split(':')[-1]
                }
            
            return None
            
        except Exception as e:
            print(f"[TRADINGVIEW SEARCH ERROR] {str(e)}")
            return None
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get real-time quote data for a symbol
        
        Args:
            symbol: Stock ticker (e.g., "SBIN", "AAPL")
            exchange: Exchange code (NSE, BSE, NASDAQ, NYSE, etc.)
        
        Returns:
            Dictionary with current price, market cap, P/E ratio, etc.
        """
        cache_key = f"quote_{exchange}:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            full_symbol = f"{exchange}:{symbol}"
            url = f"{self.quote_url}"
            
            params = {
                'symbol': full_symbol,
                'fields': ','.join([
                    'close',
                    'change',
                    'change_percent',
                    'high',
                    'low',
                    'open',
                    'volume',
                    'market_cap_basic',
                    'price_earnings_ttm',
                    'earnings_per_share_basic_ttm',
                    'number_of_employees',
                    'sector',
                    'description',
                    'name',
                    'type',
                    'currency',
                    'exchange'
                ])
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}")
            
            data = response.json()
            
            result = {
                "symbol": symbol.upper(),
                "full_symbol": full_symbol,
                "current_price": data.get("close") or 0,
                "open": data.get("open") or 0,
                "high": data.get("high") or 0,
                "low": data.get("low") or 0,
                "change": data.get("change") or 0,
                "change_percent": data.get("change_percent") or 0,
                "volume": data.get("volume") or 0,
                "market_cap": data.get("market_cap_basic") or 0,
                "pe_ratio": data.get("price_earnings_ttm") or 0,
                "eps": data.get("earnings_per_share_basic_ttm") or 0,
                "company_name": data.get("name") or symbol,
                "description": data.get("description") or "N/A",
                "sector": data.get("sector") or "Unknown",
                "employees": data.get("number_of_employees") or 0,
                "currency": data.get("currency") or "USD",
                "exchange": data.get("exchange") or exchange,
                "timestamp": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch quote for {symbol}: {str(e)}")
    
    def get_current_price(self, symbol: str) -> Dict:
        """
        Get current stock price and basic information
        Uses TradingView search to resolve exchange and ticker
        
        Args:
            symbol: Stock ticker with optional exchange suffix (e.g., "AAPL", "SBIN.NS")
        
        Returns:
            Dictionary with current price, market cap, P/E ratio, etc.
        """
        search_result = self.search_symbol(symbol)
        if search_result:
            exch = search_result.get('exchange', '')
            tick = search_result.get('ticker', '')
            if not exch or not tick:
                raise Exception("Search result missing exchange or ticker")
            return self.get_quote(tick, exch)
        
        raise Exception(f"Unable to resolve exchange for symbol '{symbol}'")
    
    def get_technical_indicators(self, symbol: str, exchange: str = "NSE") -> Dict:
        """
        Get technical analysis indicators
        
        Args:
            symbol: Stock ticker
            exchange: Exchange code
        
        Returns:
            Dictionary with RSI, MACD, moving averages, etc.
        """
        cache_key = f"technical_{exchange}:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            full_symbol = f"{exchange}:{symbol}"
            
            params = {
                'symbol': full_symbol,
                'fields': ','.join([
                    'RSI',
                    'RSI[1]',
                    'Stoch.K',
                    'Stoch.D',
                    'MACD.macd',
                    'MACD.signal',
                    'ADX',
                    'ATR',
                    'SMA20',
                    'SMA50',
                    'SMA100',
                    'SMA200',
                    'EMA20',
                    'EMA50',
                    'EMA100',
                    'EMA200',
                    'BB.upper',
                    'BB.lower',
                ])
            }
            
            response = requests.get(self.quote_url, params=params, timeout=10)
            data = response.json()
            
            result = {
                "symbol": symbol.upper(),
                "rsi": data.get("RSI", 0),
                "macd": data.get("MACD.macd", 0),
                "macd_signal": data.get("MACD.signal", 0),
                "adx": data.get("ADX", 0),
                "atr": data.get("ATR", 0),
                "sma_20": data.get("SMA20", 0),
                "sma_50": data.get("SMA50", 0),
                "sma_200": data.get("SMA200", 0),
                "ema_20": data.get("EMA20", 0),
                "ema_50": data.get("EMA50", 0),
                "bollinger_upper": data.get("BB.upper", 0),
                "bollinger_lower": data.get("BB.lower", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[TRADINGVIEW] Technical indicators error: {str(e)}")
            return {}
    
    def get_market_overview(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get data for multiple stocks at once
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Dictionary mapping symbol to stock data
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_current_price(symbol)
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return results
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()


_tradingview_client = None

def get_tradingview_client() -> TradingViewClient:
    """Get singleton TradingView client instance"""
    global _tradingview_client
    if _tradingview_client is None:
        _tradingview_client = TradingViewClient()
    return _tradingview_client

