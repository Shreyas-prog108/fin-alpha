"""
Alpha Vantage Client
Fetches market data using the Alpha Vantage API.
"""

from datetime import datetime
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import requests


class AlphaVantageClient:
    """
    Alpha Vantage data client
    Provides quote and symbol-search functionality for stocks.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("ALPHAVANTAGE_API_KEY")
            or os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHA_VANTAGE_KEY")
            or os.getenv("API_KEY")
        )
        self.timeout = 10
        self.session = requests.Session()

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key not found. Set ALPHAVANTAGE_API_KEY in your env."
            )

    def _normalize_symbol(self, symbol: str) -> str:
        ticker = symbol.upper().strip()
        if ticker.endswith(".NSI"):
            return ticker[:-4] + ".NSE"
        if ticker.endswith(".NS"):
            return ticker[:-3] + ".NSE"
        if ticker.endswith(".BOM"):
            return ticker[:-4] + ".BSE"
        if ticker.endswith(".BO"):
            return ticker[:-3] + ".BSE"
        return ticker

    def normalize_symbol(self, symbol: str) -> str:
        """
        Public symbol normalization helper for callers that need visibility
        into the exact symbol format sent to Alpha Vantage.
        """
        return self._normalize_symbol(symbol)

    def _extract_ticker_and_exchange(self, symbol: str) -> Tuple[str, str]:
        symbol = symbol.upper()
        if "." in symbol:
            ticker, suffix = symbol.split(".", 1)
            suffix = suffix.upper()
            if suffix in {"NSE", "NSI"}:
                return ticker, "NSE"
            if suffix in {"BSE", "BOM", "BO"}:
                return ticker, "BSE"
            return ticker, suffix
        return symbol, "UNKNOWN"

    def _safe_float(self, value: Optional[str]) -> float:
        if value is None or value == "":
            return 0.0
        cleaned = str(value).replace("%", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _safe_int(self, value: Optional[str]) -> int:
        if value is None or value == "":
            return 0
        cleaned = str(value).replace(",", "").strip()
        try:
            return int(float(cleaned))
        except ValueError:
            return 0

    def _period_to_days(self, period: str) -> int:
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 1825,
        }
        return mapping.get((period or "1mo").lower(), 30)

    def _request(self, **params) -> Dict:
        self._ensure_api_key()
        payload = {
            "apikey": self.api_key,
            **params,
        }
        response = self.session.get(self.BASE_URL, params=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if "Note" in data:
            raise Exception(f"Alpha Vantage rate limit: {data['Note']}")
        if "Error Message" in data:
            raise Exception(data["Error Message"])
        if "Information" in data and "api key" in str(data["Information"]).lower():
            raise Exception(data["Information"])

        return data

    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote data for a symbol.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        data = self._request(function="GLOBAL_QUOTE", symbol=normalized_symbol)
        quote = data.get("Global Quote", {})

        if not quote:
            raise Exception(f"No quote returned for {normalized_symbol}")

        resolved_symbol = quote.get("01. symbol", normalized_symbol).upper()
        ticker, exchange = self._extract_ticker_and_exchange(resolved_symbol)
        is_indian = exchange in {"NSE", "BSE"}

        return {
            "symbol": resolved_symbol,
            "current_price": self._safe_float(quote.get("05. price")),
            "open": self._safe_float(quote.get("02. open")),
            "high": self._safe_float(quote.get("03. high")),
            "low": self._safe_float(quote.get("04. low")),
            "previous_close": self._safe_float(quote.get("08. previous close")),
            "change": self._safe_float(quote.get("09. change")),
            "change_percent": round(self._safe_float(quote.get("10. change percent")), 2),
            "volume": self._safe_int(quote.get("06. volume")),
            "company_name": ticker,
            "exchange": exchange,
            "currency": "INR" if is_indian else "USD",
            "timestamp": datetime.now().isoformat(),
            "source": "alphavantage",
        }

    def get_current_price(self, symbol: str) -> Dict:
        """Alias for get_quote consistent with existing interface."""
        return self.get_quote(symbol)

    def get_technical_indicators(self, symbol: str) -> Dict:
        """
        Return lightweight technical indicators from Alpha Vantage.
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            data = self._request(
                function="RSI",
                symbol=normalized_symbol,
                interval="daily",
                time_period=14,
                series_type="close",
            )
            technical = data.get("Technical Analysis: RSI", {})
            if not technical:
                return {}
            latest_date = max(technical.keys())
            return {
                "symbol": normalized_symbol,
                "rsi": self._safe_float(technical[latest_date].get("RSI")),
                "timestamp": datetime.now().isoformat(),
                "source": "alphavantage",
            }
        except Exception as e:
            print(f"[ALPHAVANTAGE] Technical indicators error: {str(e)}")
            return {}

    def get_historical_data(self, symbol: str, period: str = "1mo") -> List[Dict]:
        """
        Get daily OHLCV candles for a period.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        days = self._period_to_days(period)
        outputsize = "full" if days > 100 else "compact"

        data = self._request(
            function="TIME_SERIES_DAILY_ADJUSTED",
            symbol=normalized_symbol,
            outputsize=outputsize,
        )
        series = data.get("Time Series (Daily)", {})
        if not series:
            data = self._request(
                function="TIME_SERIES_DAILY",
                symbol=normalized_symbol,
                outputsize=outputsize,
            )
            series = data.get("Time Series (Daily)", {})

        if not series:
            raise Exception(f"No historical series returned for {normalized_symbol}")

        rows: List[Dict] = []
        for date, point in series.items():
            rows.append(
                {
                    "date": date,
                    "time": date,
                    "open": self._safe_float(point.get("1. open")),
                    "high": self._safe_float(point.get("2. high")),
                    "low": self._safe_float(point.get("3. low")),
                    "close": self._safe_float(point.get("4. close")),
                    "volume": self._safe_int(point.get("6. volume", point.get("5. volume"))),
                    "source": "alphavantage",
                }
            )

        rows.sort(key=lambda row: row["date"])
        return rows[-days:]

    def calculate_volatility(self, symbol: str, period: str = "1mo") -> float:
        """
        Calculate annualized volatility from daily log returns.
        """
        candles = self.get_historical_data(symbol, period)
        closes = [float(row["close"]) for row in candles if row.get("close", 0) > 0]
        if len(closes) < 2:
            return 0.0

        log_returns: List[float] = []
        for idx in range(1, len(closes)):
            prev = closes[idx - 1]
            curr = closes[idx]
            if prev > 0 and curr > 0:
                log_returns.append(math.log(curr / prev))

        if len(log_returns) < 2:
            return 0.0

        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol = math.sqrt(max(variance, 0.0))
        return float(daily_vol * math.sqrt(252))

    def search_symbol(self, query: str) -> Dict:
        """
        Search for symbol using Alpha Vantage SYMBOL_SEARCH.
        """
        data = self._request(function="SYMBOL_SEARCH", keywords=query)
        matches = data.get("bestMatches", [])
        if not matches:
            return {}

        query_upper = (query or "").strip().upper()
        query_tokens = [
            token.lower()
            for token in re.findall(r"[A-Z0-9]+", query_upper)
            if len(token) >= 2
        ]
        query_suffix = query_upper.split(".", 1)[1] if "." in query_upper else ""

        def _score(item: Dict) -> float:
            symbol = item.get("1. symbol", "").upper()
            name = item.get("2. name", "").lower()
            region = item.get("4. region", "").lower()
            score = self._safe_float(item.get("9. matchScore"))

            if query_upper and symbol == query_upper:
                score += 2.0
            elif query_upper and symbol.startswith(query_upper):
                score += 1.0

            if query_suffix and symbol.endswith(f".{query_suffix}"):
                score += 0.5

            token_hits = sum(1 for token in query_tokens if token in name or token in symbol.lower())
            score += token_hits * 0.08

            # Mild preference for equity-like instruments over unrelated funds/ETNs.
            instrument_type = item.get("3. type", "").lower()
            if instrument_type in {"equity", "stock", "etf"}:
                score += 0.03

            # Small tie-breaker on region mention overlap.
            if query_tokens and any(token in region for token in query_tokens):
                score += 0.02

            return score

        best = max(matches, key=_score)

        full_symbol = best.get("1. symbol", "").upper()
        ticker, exchange = self._extract_ticker_and_exchange(full_symbol)

        return {
            "symbol": full_symbol,
            "ticker": ticker,
            "name": best.get("2. name", full_symbol),
            "exchange": exchange,
            "source": "alphavantage",
            "type": best.get("3. type", "stock"),
            "region": best.get("4. region", ""),
            "currency": best.get("8. currency", ""),
            "match_score": self._safe_float(best.get("9. matchScore")),
        }


_alphavantage_client = None


def get_alphavantage_client() -> AlphaVantageClient:
    """Get singleton Alpha Vantage client instance."""
    global _alphavantage_client
    if _alphavantage_client is None:
        _alphavantage_client = AlphaVantageClient()
    return _alphavantage_client
