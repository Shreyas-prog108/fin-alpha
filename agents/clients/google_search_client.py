"""
Google Search Client
Uses Gemini with Google Search grounding to fetch real-time financial news.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(".env.local", override=False)
load_dotenv(override=False)


class GoogleSearchClient:
    """
    News client backed by Gemini's Google Search grounding tool.
    Returns article lists in the same schema as the other news clients.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set — Google Search client unavailable."
            )

    def _build_gemini_client(self):  # returns a GenerativeModel
        try:
            # Try new google.genai package
            import google.genai as genai_client
            import google.genai.types as genai_types
            
            genai_client.configure(api_key=self.api_key)
            
            return genai_client.GenerativeModel(
                model_name=self.model,
                generation_config=genai_types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=2000,
                ),
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )
        except ImportError:
            # Fallback to older google.generativeai
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            return genai.GenerativeModel(
                model_name=self.model,
                tools=[{"google_search": {}}]
            )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(str(value).replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return default

    def _extract_json_array(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON array from text."""
        start = text.find("[")
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
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _parse_articles(self, text: str) -> List[Dict[str, Any]]:
        """Parse a JSON array of articles from LLM output."""
        candidate = text.strip()
        # strip markdown fences
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            fragment = self._extract_json_array(candidate)
            if not fragment:
                return []
            try:
                parsed = json.loads(fragment)
            except json.JSONDecodeError:
                return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("articles", "news", "results", "items"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
        return []

    def _normalize_article(
        self,
        item: Dict[str, Any],
        grounding_sources: List[Dict[str, str]],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        title = str(item.get("title", "")).strip()
        if not title:
            return None
        url = str(item.get("url", item.get("link", ""))).strip()
        # Backfill missing URLs from grounding metadata
        if not url and idx < len(grounding_sources):
            url = grounding_sources[idx].get("uri", "")
        source = str(item.get("source", "")).strip()
        if not source and idx < len(grounding_sources):
            source = grounding_sources[idx].get("title", "Google Search")
        source = source or "Google Search"
        sentiment = str(item.get("sentiment", "neutral")).lower().strip()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        return {
            "title": title,
            "description": str(item.get("description", item.get("summary", ""))).strip(),
            "url": url,
            "published_at": str(
                item.get("published_at", item.get("publishedAt", item.get("date", "")))
            ).strip(),
            "source": source,
            "sentiment": sentiment,
            "sentiment_score": max(
                -1.0, min(1.0, self._safe_float(item.get("sentiment_score")))
            ),
        }

    def _extract_grounding_sources(self, response) -> List[Dict[str, str]]:
        """Pull web sources from Gemini grounding metadata."""
        sources: List[Dict[str, str]] = []
        try:
            for candidate in response.candidates:
                gm = getattr(candidate, "grounding_metadata", None)
                if gm is None:
                    continue
                chunks = getattr(gm, "grounding_chunks", None) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        sources.append(
                            {
                                "uri": getattr(web, "uri", "") or "",
                                "title": getattr(web, "title", "") or "",
                            }
                        )
        except Exception:
            pass
        return sources

    def _run_grounded_query(self, prompt: str) -> tuple[str, List[Dict[str, str]]]:
        """Send prompt to Gemini with Google Search grounding. Returns (text, sources)."""
        self._ensure_api_key()
        gemini = self._build_gemini_client()
        response = gemini.generate_content(prompt)
        text = response.text or ""
        sources = self._extract_grounding_sources(response)
        return text, sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_news(
        self,
        symbol: str,
        company_name: str,
        days: int = 7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent stock-specific news via Gemini + Google Search grounding.
        """
        prompt = f"""Search Google for recent news about {company_name} ({symbol}) from the last {days} days.
Return ONLY a valid JSON array of up to {limit} news articles.
Each item must have these fields:
{{
  "title": "string",
  "description": "string",
  "url": "https://...",
  "published_at": "YYYY-MM-DD",
  "source": "string",
  "sentiment": "positive|negative|neutral",
  "sentiment_score": number between -1.0 and 1.0
}}
Only include articles directly relevant to {company_name} or its stock ({symbol}).
No extra text outside the JSON array."""

        text, sources = self._run_grounded_query(prompt)
        raw_articles = self._parse_articles(text)
        articles: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_articles[:limit]):
            normalized = self._normalize_article(item, sources, idx)
            if normalized:
                articles.append(normalized)
        return articles

    def get_market_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch broad market news via Gemini + Google Search grounding.
        """
        prompt = f"""Search Google for the latest global stock market news from today.
Return ONLY a valid JSON array of up to {limit} news articles.
Each item must have these fields:
{{
  "title": "string",
  "description": "string",
  "url": "https://...",
  "published_at": "YYYY-MM-DD",
  "source": "string",
  "sentiment": "positive|negative|neutral",
  "sentiment_score": number between -1.0 and 1.0
}}
No extra text outside the JSON array."""

        text, sources = self._run_grounded_query(prompt)
        raw_articles = self._parse_articles(text)
        articles: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_articles[:limit]):
            normalized = self._normalize_article(item, sources, idx)
            if normalized:
                articles.append(normalized)
        return articles

    def grounded_query(self, query: str) -> Dict[str, Any]:
        """
        Run an arbitrary grounded query and return text + sources.
        Matches the interface of PerplexityClient.quick_query().
        """
        text, sources = self._run_grounded_query(query)
        return {
            "response": text,
            "sources": [s.get("uri", "") for s in sources if s.get("uri")],
            "search_used": True,
            "model": self.model,
        }


_google_search_client: Optional[GoogleSearchClient] = None


def get_google_search_client() -> GoogleSearchClient:
    """Get singleton GoogleSearchClient instance."""
    global _google_search_client
    if _google_search_client is None:
        _google_search_client = GoogleSearchClient()
    return _google_search_client
