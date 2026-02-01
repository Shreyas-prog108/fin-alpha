"""
TradingView Client
Fetches real-time market data using tradingview-ta (unofficial API)
"""

from tradingview_ta import TA_Handler, Interval, Exchange
import requests
from typing import Dict, List, Optional
from datetime import datetime

class TradingViewClient:
    """
    TradingView data client
    Provides real-time quotes and technical analysis using tradingview-ta
    """
    
    def __init__(self):
        pass

    
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
            
            return result
            
        except Exception as e:
            print(f"[TRADINGVIEW] Technical indicators error: {str(e)}")
            return {}
            


    def search_symbol(self, query: str) -> Dict:
        """
        Search for symbol using TradingView public API (v3)
        """
        try:
            url = "https://symbol-search.tradingview.com/symbol_search/v3/"
            params = {
                "text": query,
                "hl": "1",
                "exchange": "",
                "lang": "en",
                "search_type": "stocks",
                "domain": "production"
            }
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if not data:
                return {}
            
            # Prefer NSE/BSE
            best = data[0]
            for item in data:
                ex = item.get("exchange", "")
                if ex in ["NSE", "BSE"]:
                    best = item
                    break
            
            symbol = best["symbol"]
            exchange = best.get("exchange", "NSE")
            description = best.get("description", symbol)
            
            # Format
            if exchange == "NSE":
                full_symbol = f"{symbol}.NSE"
            elif exchange == "BSE":
                full_symbol = f"{symbol}.BO"
            else:
                 full_symbol = symbol
            
            return {
                "symbol": full_symbol,
                "ticker": symbol,
                "name": description,
                "exchange": exchange,
                "source": "tradingview",
                "type": best.get("type", "stock")
            }
        except Exception as e:
            print(f"[TRADINGVIEW SEARCH ERROR] {str(e)}")
            return {}
_tradingview_client = None

def get_tradingview_client() -> TradingViewClient:
    """Get singleton TradingView client instance"""
    global _tradingview_client
    if _tradingview_client is None:
        _tradingview_client = TradingViewClient()
    return _tradingview_client

