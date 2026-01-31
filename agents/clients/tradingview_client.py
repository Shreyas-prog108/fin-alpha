"""
TradingView Client
Fetches real-time market data using tradingview-ta (unofficial API)
"""

from tradingview_ta import TA_Handler, Interval, Exchange
from typing import Dict, List, Optional
from datetime import datetime

class TradingViewClient:
    """
    TradingView data client
    Provides real-time quotes and technical analysis using tradingview-ta
    """
    
    def __init__(self):
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
    
    def _convert_symbol(self, symbol: str) -> tuple[str, str]:
        """
        Convert symbol to TradingView format (SCREENER, EXCHANGE, SYMBOL)
        Handles .NS -> NSE, .BO -> BSE conversion
        """
        screener = "india"
        exchange = "NSE"  
        ticker = symbol.upper()
        
        if ".NSE" in ticker:
            ticker = ticker.replace(".NSE", "")
            exchange = "NSE"
        elif ".BSE" in ticker:
            ticker = ticker.replace(".BSE", "")
            exchange = "BSE"
        elif ".NS" in ticker:
            ticker = ticker.replace(".NS", "")
            exchange = "NSE"
        elif ".BO" in ticker:
            ticker = ticker.replace(".BO", "")
            exchange = "BSE"
            
        return screener, exchange, ticker

    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote data for a symbol
        
        Args:
            symbol: Stock ticker (e.g., "SBIN.NS", "RELIANCE.BO")
        
        Returns:
            Dictionary with current price and other metrics
        """
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            screener, exchange, ticker = self._convert_symbol(symbol)
            
            handler = TA_Handler(
                symbol=ticker,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            
            # Extract data
            current_price = analysis.indicators.get("close", 0)
            open_price = analysis.indicators.get("open", 0)
            high = analysis.indicators.get("high", 0)
            low = analysis.indicators.get("low", 0)
            volume = analysis.indicators.get("volume", 0)
            change = analysis.indicators.get("change", 0)
            
            # Calculate change percent
            prev_close = current_price - change 
            change_percent = (change / prev_close * 100) if prev_close else 0

            result = {
                "symbol": symbol.upper(),
                "current_price": current_price,
                "open": open_price,
                "high": high,
                "low": low,
                "change": change,
                "change_percent": round(change_percent, 2),
                "volume": volume,
                "company_name": ticker, 
                "exchange": exchange,
                "currency": "INR", 
                "timestamp": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch quote for {symbol}: {str(e)}")
    
    def get_current_price(self, symbol: str) -> Dict:
        """Alias for get_quote consistent with interface"""
        return self.get_quote(symbol)
    
    def get_technical_indicators(self, symbol: str) -> Dict:
        """
        Get technical analysis indicators
        """
        cache_key = f"technical_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            screener, exchange, ticker = self._convert_symbol(symbol)
            
            handler = TA_Handler(
                symbol=ticker,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            
            result = {
                "symbol": symbol.upper(),
                "rsi": indicators.get("RSI", 0),
                "macd": indicators.get("MACD.macd", 0),
                "macd_signal": indicators.get("MACD.signal", 0),
                "adx": indicators.get("ADX", 0),
                "atr": indicators.get("ATR", 0),
                "sma_20": indicators.get("SMA20", 0),
                "sma_50": indicators.get("SMA50", 0),
                "sma_200": indicators.get("SMA200", 0),
                "ema_20": indicators.get("EMA20", 0),
                "ema_50": indicators.get("EMA50", 0),
                "bollinger_upper": indicators.get("BB.upper", 0),
                "bollinger_lower": indicators.get("BB.lower", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[TRADINGVIEW] Technical indicators error: {str(e)}")
            return {}
            
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

