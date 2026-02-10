"""
Perplexity Client
Fetches stock data and news using Perplexity's chat completions API.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(".env.local", override=False)
load_dotenv(override=False)


class PerplexityClient:
    """
    Perplexity data client for grounded stock/news retrieval.
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("PERPLEXITY_API_KEY")
            or os.getenv("PPLX_API_KEY")
        )
        self.model = model or os.getenv("PERPLEXITY_MODEL", "sonar")
        self.timeout = float(os.getenv("PERPLEXITY_TIMEOUT", "45"))
        self.session = requests.Session()

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Perplexity API key not found. Set PERPLEXITY_API_KEY in your env."
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        self._ensure_api_key()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self.session.post(
            self.BASE_URL,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "") or ""

        citations = data.get("citations") or []
        if not citations and choices:
            message = choices[0].get("message", {})
            citations = message.get("citations") or []

        return {
            "text": content,
            "citations": citations if isinstance(citations, list) else [],
            "raw": data,
        }

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        def _scan(open_char: str, close_char: str) -> Optional[str]:
            start = text.find(open_char)
            if start == -1:
                return None
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == "\"":
                        in_string = False
                    continue

                if ch == "\"":
                    in_string = True
                    continue
                if ch == open_char:
                    depth += 1
                    continue
                if ch == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            return None

        object_fragment = _scan("{", "}")
        array_fragment = _scan("[", "]")

        if object_fragment and array_fragment:
            return object_fragment if len(object_fragment) >= len(array_fragment) else array_fragment
        return object_fragment or array_fragment

    def _parse_json(self, text: str) -> Any:
        candidate = text.strip()
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fragment = self._extract_balanced_json(candidate)
            if not fragment:
                raise ValueError("Model did not return parseable JSON")
            return json.loads(fragment)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", "").replace("%", "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).replace(",", "").strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default

    def _period_to_points(self, period: str) -> int:
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

    def _normalize_symbol(self, symbol: str) -> str:
        return (symbol or "").strip().upper()

    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote and basic fundamentals for a symbol.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        prompt = f"""
Return ONLY valid JSON for the latest stock quote data for {normalized_symbol}.
Fields required:
{{
  "symbol": "{normalized_symbol}",
  "company_name": "string",
  "current_price": number,
  "open": number,
  "high": number,
  "low": number,
  "previous_close": number,
  "change": number,
  "change_percent": number,
  "volume": integer,
  "market_cap": number,
  "pe_ratio": number,
  "sector": "string",
  "industry": "string",
  "currency": "string",
  "exchange": "string"
}}
If a field is unknown, set numeric values to 0 and strings to "".
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data API. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        parsed = self._parse_json(result["text"])
        if not isinstance(parsed, dict):
            raise ValueError("Unexpected response format for stock quote")

        return {
            "symbol": self._normalize_symbol(parsed.get("symbol", normalized_symbol)),
            "company_name": parsed.get("company_name", normalized_symbol),
            "current_price": self._safe_float(parsed.get("current_price")),
            "open": self._safe_float(parsed.get("open")),
            "high": self._safe_float(parsed.get("high")),
            "low": self._safe_float(parsed.get("low")),
            "previous_close": self._safe_float(parsed.get("previous_close")),
            "change": self._safe_float(parsed.get("change")),
            "change_percent": round(self._safe_float(parsed.get("change_percent")), 2),
            "volume": self._safe_int(parsed.get("volume")),
            "market_cap": self._safe_float(parsed.get("market_cap")),
            "pe_ratio": self._safe_float(parsed.get("pe_ratio")),
            "sector": parsed.get("sector", "") or "Unknown",
            "industry": parsed.get("industry", "") or "Unknown",
            "currency": parsed.get("currency", "") or "USD",
            "exchange": parsed.get("exchange", "") or "Unknown",
            "source": "perplexity",
            "timestamp": datetime.utcnow().isoformat(),
            "citations": result.get("citations", []),
        }

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed company/fundamental info.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        prompt = f"""
Return ONLY valid JSON with detailed stock/company information for {normalized_symbol}.
Include these fields:
{{
  "symbol": "{normalized_symbol}",
  "company_name": "string",
  "exchange": "string",
  "currency": "string",
  "sector": "string",
  "industry": "string",
  "country": "string",
  "market_cap": number,
  "enterprise_value": number,
  "pe_ratio": number,
  "forward_pe": number,
  "price_to_book": number,
  "eps": number,
  "dividend_yield": number,
  "beta": number,
  "fifty_two_week_high": number,
  "fifty_two_week_low": number,
  "description": "string"
}}
Unknown numeric fields must be 0, unknown string fields must be "".
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data API. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        parsed = self._parse_json(result["text"])
        if not isinstance(parsed, dict):
            raise ValueError("Unexpected response format for stock info")
        parsed["symbol"] = self._normalize_symbol(parsed.get("symbol", normalized_symbol))
        parsed["source"] = "perplexity"
        parsed["citations"] = result.get("citations", [])
        parsed["timestamp"] = datetime.utcnow().isoformat()
        return parsed

    def get_historical_data(self, symbol: str, period: str = "1mo") -> List[Dict[str, Any]]:
        """
        Get historical OHLCV candles for period.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        points = self._period_to_points(period)
        prompt = f"""
Return ONLY valid JSON with historical DAILY candles for {normalized_symbol}.
Need around {points} most recent trading days (or all available up to that count).
Format:
{{
  "symbol": "{normalized_symbol}",
  "period": "{period}",
  "candles": [
    {{
      "date": "YYYY-MM-DD",
      "open": number,
      "high": number,
      "low": number,
      "close": number,
      "volume": integer
    }}
  ]
}}
No extra text.
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data API. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2600,
        )
        parsed = self._parse_json(result["text"])

        candles: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            raw_candles = parsed.get("candles", [])
        elif isinstance(parsed, list):
            raw_candles = parsed
        else:
            raw_candles = []

        for item in raw_candles:
            if not isinstance(item, dict):
                continue
            date_val = str(item.get("date", item.get("time", ""))).strip()
            if not date_val:
                continue
            candles.append(
                {
                    "date": date_val[:10],
                    "time": date_val[:10],
                    "open": self._safe_float(item.get("open")),
                    "high": self._safe_float(item.get("high")),
                    "low": self._safe_float(item.get("low")),
                    "close": self._safe_float(item.get("close")),
                    "volume": self._safe_int(item.get("volume")),
                    "source": "perplexity",
                }
            )

        candles.sort(key=lambda row: row.get("date", ""))
        if not candles:
            raise ValueError(f"No historical data returned for {normalized_symbol}")
        return candles[-points:]

    def calculate_volatility(self, symbol: str, period: str = "1mo") -> float:
        """
        Calculate annualized volatility from log returns.
        """
        data = self.get_historical_data(symbol, period=period)
        closes = [self._safe_float(row.get("close")) for row in data if row.get("close")]
        closes = [c for c in closes if c > 0]
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

    def get_stock_news(
        self,
        symbol: str,
        company_name: str,
        days: int = 7,
        category: str = "general",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent stock-specific news.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        safe_category = (category or "general").strip().lower()
        prompt = f"""
Return ONLY valid JSON array of recent news for {company_name} ({normalized_symbol}).
Constraints:
- Only articles from the last {days} days.
- Focus category: {safe_category}.
- Maximum {limit} articles.
Each item format:
{{
  "title": "string",
  "description": "string",
  "url": "https://...",
  "published_at": "YYYY-MM-DD",
  "source": "string",
  "sentiment": "positive|negative|neutral",
  "sentiment_score": number
}}
No extra text.
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a finance news API. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2200,
        )
        parsed = self._parse_json(result["text"])
        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                parsed = parsed.get("articles", [])
            else:
                parsed = []

        articles: List[Dict[str, Any]] = []
        for item in parsed[:limit]:
            if not isinstance(item, dict):
                continue
            sentiment = str(item.get("sentiment", "neutral")).lower().strip()
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"
            articles.append(
                {
                    "title": item.get("title", "").strip(),
                    "description": item.get("description", "").strip(),
                    "url": item.get("url", "").strip(),
                    "published_at": item.get("published_at", item.get("publishedAt", "")).strip(),
                    "source": item.get("source", "Unknown"),
                    "sentiment": sentiment,
                    "sentiment_score": max(-1.0, min(1.0, self._safe_float(item.get("sentiment_score")))),
                }
            )
        return articles

    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest broad market news.
        """
        prompt = f"""
Return ONLY valid JSON array of latest global stock market headlines.
Maximum {limit} items.
Each item:
{{
  "title": "string",
  "description": "string",
  "url": "https://...",
  "published_at": "YYYY-MM-DD",
  "source": "string",
  "sentiment": "positive|negative|neutral",
  "sentiment_score": number
}}
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a finance news API. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2200,
        )
        parsed = self._parse_json(result["text"])
        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                parsed = parsed.get("articles", [])
            else:
                parsed = []

        output: List[Dict[str, Any]] = []
        for item in parsed[:limit]:
            if not isinstance(item, dict):
                continue
            sentiment = str(item.get("sentiment", "neutral")).lower().strip()
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"
            output.append(
                {
                    "title": item.get("title", "").strip(),
                    "description": item.get("description", "").strip(),
                    "url": item.get("url", "").strip(),
                    "published_at": item.get("published_at", item.get("publishedAt", "")).strip(),
                    "source": item.get("source", "Unknown"),
                    "sentiment": sentiment,
                    "sentiment_score": max(-1.0, min(1.0, self._safe_float(item.get("sentiment_score")))),
                }
            )
        return output

    def quick_query(self, query: str) -> Dict[str, Any]:
        """
        Run a grounded query and return text + citations.
        """
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise financial research assistant.",
                },
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return {
            "response": result.get("text", ""),
            "sources": result.get("citations", []),
            "search_used": True,
            "model": self.model,
        }


_perplexity_client: Optional[PerplexityClient] = None


def get_perplexity_client() -> PerplexityClient:
    """Get singleton Perplexity client instance."""
    global _perplexity_client
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    return _perplexity_client
