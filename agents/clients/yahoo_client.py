"""
Yahoo Finance Client
Fetches real-time and historical market data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime
import time


class YahooFinanceClient:
    """
    Yahoo Finance data client
    Provides stock prices, company info, historical data, and financial metrics
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300
        self._session = self._create_session()
    
    def _create_session(self):
        """Create a session with browser-like headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        return session

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get yfinance Ticker with custom session"""
        return yf.Ticker(symbol, session=self._session)
    
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
    
    def get_current_price(self, symbol: str) -> Dict:
        """
        Get current stock price and basic information
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")
        
        Returns:
            Dictionary with current price, market cap, P/E ratio, etc.
        
        Raises:
            ValueError: If symbol is invalid
            Exception: If data fetch fails
        """
        
        cache_key = f"price_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Retry with backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = self._get_ticker(symbol)
                
                
                hist = ticker.history(period="5d")
                
                if hist.empty:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt) 
                        continue
                    raise Exception(f"No data available for {symbol}")
                
                # Get last available data
                last_row = hist.iloc[-1]
                prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else last_row['Close']
                
                current_price = float(last_row['Close'])
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0
                
                result = {
                    "symbol": symbol.upper(),
                    "current_price": round(current_price, 2),
                    "previous_close": round(prev_close, 2),
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "day_high": round(float(last_row['High']), 2),
                    "day_low": round(float(last_row['Low']), 2),
                    "volume": int(last_row['Volume']),
                    "company_name": symbol.replace('.NSE', '').replace('.BO', ''),
                    "market_cap": 0,
                    "pe_ratio": 0,
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "currency": "INR" if '.NSE' in symbol or '.BO' in symbol else "USD",
                    "exchange": "NSE" if '.NSE' in symbol else ("BSE" if '.BO' in symbol else "Unknown")
                }
                
                self._set_cache(cache_key, result)
                print(f"[YAHOO] Got price for {symbol}: {current_price}")
                return result
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    print(f"[YAHOO] Rate limit (429) hit for {symbol}. Stopping retries.")
                    raise Exception(f"Yahoo Rate Limit: {error_msg}")
                
                if attempt < max_retries - 1:
                    print(f"[YAHOO] Retry {attempt + 1} for {symbol}: {error_msg[:50]}")
                    time.sleep(2 ** attempt + 1) 
                else:
                    raise Exception(f"Failed to fetch price for {symbol}: {error_msg}")
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get comprehensive stock information and metrics
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with detailed company information and financial metrics
        """
        cache_key = f"info_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol.upper(),
                "name": info.get("longName", symbol),
                "description": info.get("longBusinessSummary", "N/A"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "total_revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "payout_ratio": info.get("payoutRatio", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0)
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch info for {symbol}: {str(e)}")
    
    def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo"
    ) -> List[Dict]:
        """
        Get historical price and volume data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
        
        Returns:
            List of dictionaries with OHLCV data
        """
        try:
            ticker = self._get_ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return []
            
            result = []
            for index, row in hist.iterrows():
                result.append({
                    "date": index.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch historical data for {symbol}: {str(e)}")
    
    def get_financial_metrics(self, symbol: str) -> Dict:
        """
        Get key financial metrics and ratios
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with financial metrics
        """
        cache_key = f"metrics_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol.upper(),
                "profit_margin": info.get("profitMargins", 0) * 100,
                "operating_margin": info.get("operatingMargins", 0) * 100,
                "roe": info.get("returnOnEquity", 0) * 100,
                "roa": info.get("returnOnAssets", 0) * 100,
                "roic": info.get("returnOnCapital", 0) * 100,
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "quick_ratio": info.get("quickRatio", 0),
                "interest_coverage": info.get("interestCoverage", 0),
                "revenue_growth": info.get("revenueGrowth", 0) * 100,
                "earnings_growth": info.get("earningsGrowth", 0) * 100,
                "eps_growth": info.get("earningsQuarterlyGrowth", 0) * 100,
                "asset_turnover": info.get("assetTurnover", 0),
                "inventory_turnover": info.get("inventoryTurnover", 0),
                "earnings_per_share": info.get("trailingEps", 0),
                "book_value_per_share": info.get("bookValue", 0),
                "revenue_per_share": info.get("revenuePerShare", 0),
                "operating_cash_flow": info.get("operatingCashflow", 0),
                "free_cash_flow": info.get("freeCashflow", 0),
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch metrics for {symbol}: {str(e)}")
    
    def calculate_volatility(self, symbol: str, period: str = "1mo") -> float:
        """
        Calculate annualized volatility from historical data
        
        Args:
            symbol: Stock ticker symbol
            period: Time period for calculation
        
        Returns:
            Annualized volatility (e.g., 0.25 for 25%)
        """
        try:
            hist_data = self.get_historical_data(symbol, period)
            
            if len(hist_data) < 2:
                return 0.0
            closes = [d["close"] for d in hist_data]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                      for i in range(1, len(closes))]
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252) 
            
            return float(annualized_volatility)
            
        except Exception as e:
            raise Exception(f"Failed to calculate volatility for {symbol}: {str(e)}")
    
    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Dict]:
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

_yahoo_client = None

def get_yahoo_client() -> YahooFinanceClient:
    """Get singleton Yahoo Finance client instance"""
    global _yahoo_client
    if _yahoo_client is None:
        _yahoo_client = YahooFinanceClient()
    return _yahoo_client