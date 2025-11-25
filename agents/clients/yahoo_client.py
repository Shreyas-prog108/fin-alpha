"""
Yahoo Finance Client
Fetches real-time and historical market data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class YahooFinanceClient:
    """
    Yahoo Finance data client
    Provides stock prices, company info, historical data, and financial metrics
    """
    
    def __init__(self):
        self.cache = {}  # Simple cache: {key: (data, timestamp)}
        self.cache_ttl = 300  # 5 minutes
    
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
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol.upper(),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown")
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            raise Exception(f"Failed to fetch price for {symbol}: {str(e)}")
    
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
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol.upper(),
                "name": info.get("longName", symbol),
                "description": info.get("longBusinessSummary", "N/A"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                
                # Valuation
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                
                # Profitability
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                
                # Financial Health
                "total_revenue": info.get("totalRevenue", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                
                # Dividends
                "dividend_yield": info.get("dividendYield", 0),
                "payout_ratio": info.get("payoutRatio", 0),
                
                # Trading
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
            ticker = yf.Ticker(symbol)
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
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol.upper(),
                
                # Profitability Metrics
                "profit_margin": info.get("profitMargins", 0) * 100,  # Convert to percentage
                "operating_margin": info.get("operatingMargins", 0) * 100,
                "roe": info.get("returnOnEquity", 0) * 100,
                "roa": info.get("returnOnAssets", 0) * 100,
                "roic": info.get("returnOnCapital", 0) * 100,
                
                # Valuation Ratios
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "ev_to_ebitda": info.get("enterpriseToEbitda", 0),
                
                # Financial Health
                "debt_to_equity": info.get("debtToEquity", 0),
                "current_ratio": info.get("currentRatio", 0),
                "quick_ratio": info.get("quickRatio", 0),
                "interest_coverage": info.get("interestCoverage", 0),
                
                # Growth Metrics
                "revenue_growth": info.get("revenueGrowth", 0) * 100,
                "earnings_growth": info.get("earningsGrowth", 0) * 100,
                "eps_growth": info.get("earningsQuarterlyGrowth", 0) * 100,
                
                # Efficiency Ratios
                "asset_turnover": info.get("assetTurnover", 0),
                "inventory_turnover": info.get("inventoryTurnover", 0),
                
                # Per Share Metrics
                "earnings_per_share": info.get("trailingEps", 0),
                "book_value_per_share": info.get("bookValue", 0),
                "revenue_per_share": info.get("revenuePerShare", 0),
                
                # Cash Flow
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
            
            # Extract close prices
            closes = [d["close"] for d in hist_data]
            
            # Calculate daily returns
            returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                      for i in range(1, len(closes))]
            
            # Calculate volatility (annualized)
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
            
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


# Singleton instance
_yahoo_client = None

def get_yahoo_client() -> YahooFinanceClient:
    """Get singleton Yahoo Finance client instance"""
    global _yahoo_client
    if _yahoo_client is None:
        _yahoo_client = YahooFinanceClient()
    return _yahoo_client