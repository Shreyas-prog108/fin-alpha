"""
Perplexity Client
Fetches stock data and news using Perplexity's chat completions API (via OpenAI SDK).
Falls back to Gemini (google-generativeai) when Perplexity is unavailable.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env.local", override=False)
load_dotenv(override=False)


class PerplexityClient:
    """
    Perplexity data client for grounded stock/news retrieval.
    Uses the OpenAI-compatible Perplexity API with Gemini as a fallback.
    """

    PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("PERPLEXITY_API_KEY")
            or os.getenv("PPLX_API_KEY")
        )
        self.model = model or os.getenv("PERPLEXITY_MODEL", "sonar")
        self.timeout = float(os.getenv("PERPLEXITY_TIMEOUT", "45"))

        # OpenAI SDK client pointed at Perplexity (lazy — only created when needed)
        self._openai_client: Optional[OpenAI] = None

        # Gemini fallback config
        self._google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self._gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            if not self.api_key:
                raise ValueError(
                    "Perplexity API key not found. Set PERPLEXITY_API_KEY in your env."
                )
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.PERPLEXITY_BASE_URL,
                timeout=self.timeout,
            )
        return self._openai_client

    def _chat_perplexity(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Call Perplexity via the OpenAI SDK."""
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = ""
        citations: List[Any] = []
        if response.choices:
            msg = response.choices[0].message
            content = msg.content or ""
            # Perplexity returns citations in the raw response object
            raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
            citations = raw_dict.get("citations") or []
            if not citations:
                # Some SDK versions surface them on the message object
                citations = getattr(msg, "citations", None) or []

        return {
            "text": content,
            "citations": citations if isinstance(citations, list) else [],
        }

    def _chat_gemini(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Call Gemini as a fallback via google-generativeai."""
        if not self._google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set — Gemini fallback unavailable."
            )
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self._google_api_key)
        gemini = genai.GenerativeModel(
            model_name=self._gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        # Convert OpenAI-style message list to a single prompt string
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System]: {content}")
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        gemini_response = gemini.generate_content(prompt)
        content = gemini_response.text or ""
        return {
            "text": content,
            "citations": [],
        }

    def _chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Try Perplexity first; fall back to Gemini on any exception."""
        try:
            return self._chat_perplexity(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as perplexity_error:
            try:
                result = self._chat_gemini(messages, temperature=temperature, max_tokens=max_tokens)
                result["fallback"] = "gemini"
                result["perplexity_error"] = str(perplexity_error)
                return result
            except Exception as gemini_error:
                raise RuntimeError(
                    f"Both Perplexity and Gemini failed. "
                    f"Perplexity: {perplexity_error}. Gemini: {gemini_error}"
                ) from gemini_error

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract the largest balanced JSON object/array from text."""
        
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
            # If we get here, JSON is incomplete - try to fix it
            if depth > 0:
                # Add missing closing braces
                fixed = text[start:] + (close_char * depth)
                return fixed
            return None

        object_fragment = _scan("{", "}")
        array_fragment = _scan("[", "]")

        if object_fragment and array_fragment:
            return object_fragment if len(object_fragment) >= len(array_fragment) else array_fragment
        return object_fragment or array_fragment

    def _try_fix_incomplete_json(self, text: str) -> str:
        """Try to fix incomplete JSON by adding missing braces/quotes."""
        # Find first { and try to close it
        start = text.find("{")
        if start == -1:
            return text
        
        # Count unclosed braces
        depth = 0
        in_string = False
        escaped = False
        last_pos = start
        
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
                continue
            if in_string:
                continue
                
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            last_pos = idx
        
        # Add missing closing braces
        if depth > 0:
            text = text[:last_pos+1] + ("}" * depth)
        
        return text

    def _extract_json_with_regex(self, text: str) -> Optional[Dict]:
        """Extract JSON object using regex patterns."""
        import re
        
        # Try to find JSON object pattern
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested
            r'\{[^{}]+\}',  # Single level
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    continue
        return None

    def _parse_json(self, text: str) -> Any:
        """Parse JSON from LLM response. Returns empty dict on failure instead of raising."""
        if not text or not text.strip():
            print("[PERPLEXITY] Empty response")
            return {}
            
        candidate = text.strip()
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
        
        # Method 1: Direct parse
        try:
            return json.loads(candidate)
        except:
            pass
        
        # Method 2: Fix incomplete JSON
        try:
            fixed = self._try_fix_incomplete_json(candidate)
            result = json.loads(fixed)
            if result:
                return result
        except:
            pass
        
        # Method 3: Extract balanced JSON
        fragment = self._extract_balanced_json(candidate)
        if fragment:
            try:
                return json.loads(fragment)
            except:
                # Try fixing the fragment
                fixed_fragment = self._try_fix_incomplete_json(fragment)
                try:
                    return json.loads(fixed_fragment)
                except:
                    pass
        
        # Method 4: Regex extraction
        regex_result = self._extract_json_with_regex(candidate)
        if regex_result:
            return regex_result
        
        # If all methods fail, log and return empty dict
        print(f"[PERPLEXITY] Could not parse JSON. Response preview: {text[:500]}...")
        return {}

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
        
        # Use symbol as-is for company name
        company_name = normalized_symbol.split(".")[0] if "." in normalized_symbol else normalized_symbol
        
        prompt = f"""
Return ONLY valid JSON for the latest stock quote for {company_name} (ticker: {normalized_symbol}).
Fields required (provide actual values, do not leave as null):
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
  "beta": number,
  "fifty_two_week_high": number,
  "fifty_two_week_low": number,
  "sector": "string",
  "industry": "string",
  "currency": "string",
  "exchange": "string"
}}
Output complete valid JSON only. Do not truncate.
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data API. Output complete valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
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
            "beta": self._safe_float(parsed.get("beta")),
            "fifty_two_week_high": self._safe_float(parsed.get("fifty_two_week_high")),
            "fifty_two_week_low": self._safe_float(parsed.get("fifty_two_week_low")),
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
        
        # Use symbol as-is for company name
        company_name = normalized_symbol.split(".")[0] if "." in normalized_symbol else normalized_symbol
        
        prompt = f"""
Return ONLY valid JSON with detailed stock/company information for {company_name} (ticker: {normalized_symbol}).
Include these fields (provide actual values):
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
Output complete valid JSON only.
"""
        result = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data API. Output complete valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
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
