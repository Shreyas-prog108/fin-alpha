from langchain.tools import tool
from typing import Dict, List, Optional, Tuple
import json
import re

from .clients import (
    get_backend_client,
    get_alphavantage_client,
    get_mint_client,
    get_perplexity_client,
    get_news_client,
    get_google_search_client,
)

backend = get_backend_client()
alphavantage = get_alphavantage_client()
mint_client = get_mint_client()
perplexity = get_perplexity_client()
news_api_client = get_news_client()
google_search_client = get_google_search_client()

MAX_NEWS_ARTICLES_FOR_LLM = 3

_NEWS_TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "stock", "share",
    "shares", "limited", "ltd", "inc", "corp", "company", "group", "holdings",
    "etf", "fund", "trust", "plc", "ag", "sa", "se", "nv",
}


def _canonical_symbol(symbol: str) -> str:
    return alphavantage.normalize_symbol((symbol or "").strip().upper())


def _symbols_equivalent(requested_symbol: str, returned_symbol: str) -> bool:
    requested = _canonical_symbol(requested_symbol)
    returned = _canonical_symbol(returned_symbol)
    if not requested or not returned:
        return False
    if requested == returned:
        return True
    req_base, _, req_suffix = requested.partition(".")
    ret_base, _, ret_suffix = returned.partition(".")
    if req_base != ret_base:
        return False
    if not req_suffix or not ret_suffix:
        return True
    return req_suffix == ret_suffix


def _tokenize_company_name(company_name: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", (company_name or "").lower()):
        if len(token) <= 2 or token in _NEWS_TOKEN_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _is_relevant_article(article: Dict, symbol: str, company_name: str) -> bool:
    title = str(article.get("title", ""))
    description = str(article.get("description", article.get("summary", "")))
    text_blob = f"{title} {description}".lower()
    full_symbol = (symbol or "").lower()
    symbol_base = full_symbol.split(".", 1)[0]

    if full_symbol and full_symbol in text_blob:
        return True
    if symbol_base and symbol_base in text_blob:
        return True

    company_name_lower = (company_name or "").lower().strip()
    if company_name_lower and company_name_lower in text_blob:
        return True

    tokens = _tokenize_company_name(company_name)
    if not tokens:
        return False
    token_hits = sum(1 for token in tokens if token in text_blob)
    required_hits = 2 if len(tokens) >= 2 else 1
    return token_hits >= required_hits


def _filter_relevant_articles(
    articles: List[Dict], symbol: str, company_name: str
) -> List[Dict]:
    return [
        article
        for article in articles
        if isinstance(article, dict) and _is_relevant_article(article, symbol, company_name)
    ]


def _get_historical_data(symbol: str, period: str = "1mo") -> List[Dict]:
    """Fetch historical candles using Perplexity first, then Alpha Vantage fallback."""
    try:
        return perplexity.get_historical_data(symbol, period)
    except Exception as perplexity_error:
        try:
            return alphavantage.get_historical_data(symbol, period)
        except Exception as alpha_error:
            raise Exception(
                f"Perplexity historical data failed: {str(perplexity_error)}. "
                f"Alpha Vantage fallback failed: {str(alpha_error)}"
            )


def _get_stock_price_data(symbol: str) -> Dict:
    """Fetch quote using Perplexity first (better for Indian stocks), Alpha Vantage fallback."""
    requested_symbol = (symbol or "").strip().upper()
    normalized_symbol = _canonical_symbol(requested_symbol)
    
    # Default response in case everything fails
    default_result = {
        "symbol": normalized_symbol,
        "requested_symbol": requested_symbol,
        "normalized_symbol": normalized_symbol,
        "company_name": symbol,
        "current_price": 0,
        "change": 0,
        "change_percent": 0,
        "volume": 0,
        "market_cap": 0,
        "pe_ratio": 0,
        "beta": 0,
        "fifty_two_week_high": 0,
        "fifty_two_week_low": 0,
        "sector": "Unknown",
        "industry": "Unknown",
        "currency": "INR" if ".NSE" in symbol.upper() or ".BSE" in symbol.upper() else "USD",
        "exchange": "NSE" if ".NSE" in symbol.upper() else ("BSE" if ".BSE" in symbol.upper() else "Unknown"),
    }
    
    try:
        # Primary: Perplexity (better for Indian stocks)
        result = perplexity.get_stock_price(symbol)
        if result and isinstance(result, dict) and result.get("current_price", 0) > 0:
            result["source"] = "perplexity_primary"
            result["requested_symbol"] = requested_symbol
            result["normalized_symbol"] = normalized_symbol
            return result
    except Exception as e:
        print(f"[PRICE] Perplexity failed: {e}")
        
    # Fallback: Alpha Vantage (may be rate limited)
    try:
        result = alphavantage.get_current_price(normalized_symbol)
        result["source"] = "alphavantage_fallback"
        result["requested_symbol"] = requested_symbol
        result["normalized_symbol"] = normalized_symbol
        return result
    except Exception as e:
        print(f"[PRICE] Alpha Vantage failed: {e}")
    
    # Last resort: return default
    default_result["source"] = "default_fallback"
    return default_result


def _enhance_with_llm(data: Dict, symbol: str) -> Dict:
    """Enhance stock data with LLM for missing fields."""
    try:
        from .clients.perplexity_client import get_perplexity_client
        pplx = get_perplexity_client()
        
        prompt = f"""
Search for the latest stock data for {symbol.upper()}.
Return ONLY valid JSON with these fields if available:
{{
  "company_name": "string",
  "market_cap": number,
  "pe_ratio": number,
  "beta": number,
  "fifty_two_week_high": number,
  "fifty_two_week_low": number,
  "sector": "string",
  "industry": "string"
}}
Use null for unavailable data. No extra text.
"""
        pplx_result = pplx._chat(
            messages=[
                {"role": "system", "content": "You are a financial data API. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        llm_data = pplx._parse_json(pplx_result["text"])
        if isinstance(llm_data, dict):
            for k, v in llm_data.items():
                if v is not None and v != "" and v != 0:
                    # Only fill in missing fields
                    if not data.get(k) or data.get(k) == 0:
                        data[k] = pplx._safe_float(v) if isinstance(v, (int, float, str)) else v
    except Exception as e:
        # Don't fail the whole process - just log and continue
        print(f"[LLM ENHANCE] Failed: {e}")
        
        # Try fallback: extract data from text response directly
        try:
            from .clients.perplexity_client import get_perplexity_client
            pplx = get_perplexity_client()
            
            # Try a simpler approach - just get the text and try to extract
            prompt = f"""
Provide the following stock data for {symbol.upper()} in plain text format (not JSON):
- Company Name:
- Market Cap: 
- P/E Ratio:
- Beta:
- 52-Week High:
- 52-Week Low:
- Sector:
- Industry:
"""
            pplx_result = pplx._chat(
                messages=[
                    {"role": "system", "content": "You are a financial data assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            # Try to parse numbers from the text response
            text = pplx_result.get("text", "")
            import re
            
            # Extract market cap patterns like "5.2 trillion", "500 billion"
            mc_match = re.search(r'(?:market\s*cap[:\s]*)([\d,\.]+)\s*(trillion|billion|million|crore|lakh)?', text, re.IGNORECASE)
            if mc_match and (not data.get("market_cap") or data.get("market_cap") == 0):
                val = float(mc_match.group(1).replace(",", ""))
                unit = mc_match.group(2).lower() if mc_match.group(2) else ""
                if unit == "trillion":
                    val *= 1e12
                elif unit == "billion":
                    val *= 1e9
                elif unit == "million":
                    val *= 1e6
                elif unit == "crore":
                    val *= 1e7
                elif unit == "lakh":
                    val *= 1e5
                data["market_cap"] = val
            
            # Extract sector
            sector_match = re.search(r'sector[:\s]*([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
            if sector_match and (not data.get("sector") or data.get("sector") == "Unknown"):
                data["sector"] = sector_match.group(1).strip()
            
            # Extract industry
            industry_match = re.search(r'industry[:\s]*([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
            if industry_match and (not data.get("industry") or data.get("industry") == "Unknown"):
                data["industry"] = industry_match.group(1).strip()
                
        except Exception as fallback_error:
            print(f"[LLM ENHANCE] Text fallback also failed: {fallback_error}")
        
    return data


def _get_stock_info_data(symbol: str) -> Dict:
    """Fetch detailed info using Perplexity first (better for Indian stocks), Alpha Vantage fallback."""
    requested_symbol = (symbol or "").strip().upper()
    normalized_symbol = _canonical_symbol(requested_symbol)
    
    try:
        # Primary: Perplexity (better for Indian stocks)
        result = perplexity.get_stock_info(symbol)
        returned_symbol = result.get("symbol", requested_symbol)
        if not _symbols_equivalent(normalized_symbol, returned_symbol):
            raise Exception(
                "Perplexity symbol mismatch "
                f"(requested={normalized_symbol}, returned={returned_symbol})"
            )
        result["source"] = "perplexity_primary"
        result["requested_symbol"] = requested_symbol
        result["normalized_symbol"] = normalized_symbol
        
        # If Perplexity returned good data, use it
        if result.get("company_name"):
            return result
        
        raise Exception("Perplexity returned no useful data")
        
    except Exception as perplexity_error:
        # Fallback: Alpha Vantage
        try:
            fallback_quote = alphavantage.get_current_price(normalized_symbol)
            result = {
                "symbol": normalized_symbol,
                "company_name": fallback_quote.get("company_name", normalized_symbol),
                "exchange": fallback_quote.get("exchange", ""),
                "currency": fallback_quote.get("currency", ""),
                "sector": "",
                "industry": "",
                "country": "",
                "market_cap": fallback_quote.get("market_cap", 0),
                "enterprise_value": 0,
                "pe_ratio": fallback_quote.get("pe_ratio", 0),
                "forward_pe": 0,
                "price_to_book": 0,
                "eps": 0,
                "dividend_yield": 0,
                "beta": fallback_quote.get("beta", 0),
                "fifty_two_week_high": fallback_quote.get("fifty_two_week_high", 0),
                "fifty_two_week_low": fallback_quote.get("fifty_two_week_low", 0),
                "description": "",
                "source": "alphavantage_fallback",
                "requested_symbol": requested_symbol,
                "normalized_symbol": normalized_symbol,
                "warning": f"Perplexity failed: {str(perplexity_error)}",
            }
            return result
        except Exception as alpha_error:
            return {
                "symbol": normalized_symbol,
                "company_name": normalized_symbol,
                "source": "failed",
                "error": f"Perplexity: {str(perplexity_error)}, Alpha Vantage: {str(alpha_error)}"
            }


def _get_stock_news_data(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
    limit: int = 10,
) -> List[Dict]:
    """Fetch stock news — LiveMint (primary) → Google Search (secondary) → NewsAPI (fallback)."""
    mint_articles: List[Dict] = []
    google_articles: List[Dict] = []
    newsapi_articles: List[Dict] = []
    errors = []

    # Primary: LiveMint
    try:
        mint_articles = mint_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            limit=limit,
        )
    except Exception as e:
        errors.append(f"LiveMint: {str(e)}")

    # Secondary: Google Search (Gemini grounding)
    try:
        google_articles = google_search_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            limit=limit,
        )
    except Exception as e:
        errors.append(f"GoogleSearch: {str(e)}")

    # Fallback: NewsAPI
    try:
        newsapi_articles = news_api_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
        )[: limit * 2]
    except Exception as e:
        errors.append(f"NewsAPI: {str(e)}")

    mint_articles = _filter_relevant_articles(mint_articles, symbol, company_name)
    google_articles = _filter_relevant_articles(google_articles, symbol, company_name)
    newsapi_articles = _filter_relevant_articles(newsapi_articles, symbol, company_name)

    merged = _merge_news_articles(
        [
            *mint_articles,
            *google_articles,
            *newsapi_articles,
        ],
        limit=limit,
    )
    if not merged and errors:
        raise Exception(" | ".join(errors))
    return merged


def _get_newsapi_stock_news(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
    limit: int = 10,
) -> List[Dict]:
    """Fetch stock news from NewsAPI only."""
    try:
        raw_articles = news_api_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
        )
        return _merge_news_articles(
            _filter_relevant_articles(raw_articles, symbol, company_name),
            limit=limit,
        )
    except Exception:
        return []


def _normalize_news_article(article: Dict, fallback_source: str = "Unknown") -> Dict:
    title = str(article.get("title", "")).strip()
    source = str(article.get("source", fallback_source)).strip() or fallback_source
    try:
        sentiment_score = float(article.get("sentiment_score", 0) or 0)
    except (TypeError, ValueError):
        sentiment_score = 0.0
    return {
        "title": title,
        "description": str(article.get("description", article.get("summary", ""))).strip(),
        "url": str(article.get("url", article.get("link", ""))).strip(),
        "published_at": str(article.get("published_at", article.get("publishedAt", article.get("date", "")))).strip(),
        "source": source,
        "sentiment": str(article.get("sentiment", "neutral")).strip().lower() or "neutral",
        "sentiment_score": sentiment_score,
    }


def _merge_news_articles(articles: List[Dict], limit: int = 20) -> List[Dict]:
    """Normalize + de-duplicate news articles across sources."""
    merged: List[Dict] = []
    seen = set()
    for raw in articles:
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_news_article(raw)
        if not normalized["title"]:
            continue
        key = (
            normalized["url"].lower().strip()
            if normalized["url"]
            else f"{normalized['title'].lower().strip()}::{normalized['source'].lower().strip()}"
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)

    # Prefer items with timestamp first.
    merged.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return merged[:limit]


def _timeframe_to_days(time_frame: str) -> int:
    mapping = {
        "1d": 1,
        "5d": 5,
        "1wk": 7,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
    }
    return mapping.get((time_frame or "1mo").lower(), 30)


def _build_short_stock_analysis(
    *,
    symbol: str,
    company_name: str,
    price_data: Optional[Dict],
    news_articles: List[Dict],
) -> str:
    if not price_data:
        return f"{company_name} ({symbol}): stock snapshot unavailable."

    current_price = price_data.get("current_price", 0)
    change_percent = price_data.get("change_percent", 0)
    currency = price_data.get("currency", "USD")
    sentiment_votes = [a.get("sentiment", "neutral") for a in news_articles]
    positive = sentiment_votes.count("positive")
    negative = sentiment_votes.count("negative")
    neutral = sentiment_votes.count("neutral")

    if positive > negative:
        sentiment = "positive"
    elif negative > positive:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    headline = news_articles[0].get("title", "No major recent headline") if news_articles else "No major recent headline"
    return (
        f"{company_name} ({symbol}) is trading at {current_price} {currency} "
        f"({change_percent}% move). Recent news sentiment is {sentiment} "
        f"(+{positive}/-{negative}/~{neutral}); key headline: {headline}."
    )


def _resolve_symbol_from_query(query: str) -> Tuple[Optional[str], Optional[str], Dict]:
    """
    Resolve a symbol/company from free-form query via Alpha Vantage search.
    Returns (symbol, company_name, symbol_debug_info).
    """
    try:
        result = alphavantage.search_symbol(query)
        if not result:
            return None, None, {}
        match_score = float(result.get("match_score", 0) or 0)
        if match_score < 0.2:
            return None, None, {
                "requested_query": query,
                "resolver": "alphavantage_search",
                "match_score": match_score,
                "note": "No confident symbol match found.",
            }
        raw_symbol = result.get("symbol", "") or result.get("ticker", "")
        normalized_symbol = alphavantage.normalize_symbol(raw_symbol)
        company_name = result.get("name", normalized_symbol) or normalized_symbol
        symbol_debug = {
            "requested_query": query,
            "resolved_symbol": raw_symbol,
            "alphavantage_symbol": normalized_symbol,
            "exchange": result.get("exchange", ""),
            "match_score": match_score,
            "resolver": "alphavantage_search",
        }
        return normalized_symbol, company_name, symbol_debug
    except Exception:
        return None, None, {}


# TOOL-1: STOCK-PRICE
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price using Perplexity (fallback to Alpha Vantage)."""
    try:
        result = _get_stock_price_data(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch stock price: {str(e)}"})


# TOOL-2: STOCK-INFO
@tool
def get_stock_info(symbol: str) -> str:
    """Get detailed stock info using Perplexity."""
    try:
        result = _get_stock_info_data(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# TOOL-3: HISTORICAL-DATA
@tool
def get_hist_data(symbol: str, period: str = "1mo") -> str:
    """Get historical OHLCV data using Perplexity (fallback to Alpha Vantage)."""
    try:
        result = _get_historical_data(symbol, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch historical data: {str(e)}"})


# TOOL-4: ANALYZE-RISK
@tool
async def get_analyze_risk(symbol: str, period: str = "1mo") -> str:
    """Get comprehensive risk analysis using historical data."""
    try:
        hist_data = _get_historical_data(symbol, period)
        if not hist_data:
            return json.dumps({"error": "No historical data available"})

        risk_payload = [
            {
                "time": row.get("date") or row.get("time"),
                "close": row.get("close"),
                "volume": row.get("volume", 0),
            }
            for row in hist_data
        ]

        result = await backend.analyze_risk(symbol, risk_payload)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Risk analysis failed: {str(e)}"})


# TOOL-5: PREDICT-PRICE
@tool
async def predict_price(symbol: str, method: str = "ema", period: str = "1mo") -> str:
    """Predict stock price using historical data."""
    try:
        hist_data = _get_historical_data(symbol, period)
        if not hist_data:
            return json.dumps({"error": "No historical data available"})

        result = await backend.predict_price(symbol, hist_data, method=method)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Price prediction failed: {str(e)}"})


# TOOL-6: GET-MARKET-MAKER-QUOTE
@tool
async def get_market_maker_quote(symbol: str, risk_aversion: float = 0.1) -> str:
    """Calculate optimal bid/ask using Avellaneda-Stoikov."""
    try:
        price_data = _get_stock_price_data(symbol)
        mid_price = float(price_data.get("current_price", 0))
        if mid_price <= 0:
            return json.dumps({"error": "Unable to determine valid mid price"})

        try:
            volatility = float(perplexity.calculate_volatility(symbol, period="1mo"))
        except Exception:
            volatility = float(alphavantage.calculate_volatility(symbol, period="1mo"))

        if volatility <= 0:
            volatility = 0.01

        result = await backend.get_market_maker_quote(
            mid_price=mid_price,
            volatility=volatility,
            risk_aversion=risk_aversion,
        )
        result["symbol"] = symbol
        result["volatility_used"] = volatility
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Market maker quote failed: {str(e)}"})


# TOOL-7: GET-STOCK-NEWS
@tool
def get_stock_news(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
) -> str:
    """Get stock news using Perplexity."""
    try:
        articles = _get_stock_news_data(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )
        return json.dumps({"articles": articles, "count": len(articles)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})


# TOOL-8: ANALYZE-SENTIMENT
@tool
def analyze_news_sentiment(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
) -> str:
    """Analyze sentiment from recent Perplexity news results."""
    try:
        articles = _get_stock_news_data(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )
        if not articles:
            return json.dumps({"sentiment": "neutral", "reason": "No recent news found"})

        sentiments = [a.get("sentiment", "neutral") for a in articles]
        scores = [a.get("sentiment_score", 0) for a in articles]

        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        neutral = sentiments.count("neutral")
        avg_score = sum(scores) / len(scores) if scores else 0

        overall = "positive" if positive > negative else ("negative" if negative > positive else "neutral")

        return json.dumps(
            {
                "overall_sentiment": overall,
                "average_score": round(avg_score, 2),
                "sentiment_score": round(avg_score, 2),
                "breakdown": {"positive": positive, "negative": negative, "neutral": neutral},
                "article_count": len(articles),
                "total_articles": len(articles),
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": neutral,
                "confidence": "medium",
                "summary": f"Based on {len(articles)} recent articles.",
                "top_headlines": [a.get("title", "") for a in articles[:MAX_NEWS_ARTICLES_FOR_LLM]],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"Sentiment analysis failed: {str(e)}"})


# TOOL-9: GET-MARKET-NEWS
@tool
def get_market_news(limit: int = 10) -> str:
    """Get broad market news using LiveMint + Google Search."""
    try:
        mint_articles = mint_client.get_market_news(limit=limit) if hasattr(mint_client, "get_market_news") else []
        google_articles = google_search_client.get_market_news(limit=limit)
        articles = _merge_news_articles(mint_articles + google_articles, limit=limit)
        return json.dumps({"articles": articles, "count": len(articles)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch market news: {str(e)}"})


# TOOL-10: GET-FINANCIAL-METRICS
@tool
def get_financial_metrics(symbol: str) -> str:
    """Get comprehensive financial metrics including profitability, valuation, growth, and leverage ratios."""
    try:
        from .clients.perplexity_client import get_perplexity_client
        pplx = get_perplexity_client()
        
        # Try to get basic data - but don't fail completely if these don't work
        try:
            info = _get_stock_info_data(symbol)
        except Exception as e:
            print(f"[FINANCIAL METRICS] Info fetch failed: {e}")
            info = {}
        
        try:
            price_data = _get_stock_price_data(symbol)
        except Exception as e:
            print(f"[FINANCIAL METRICS] Price fetch failed: {e}")
            price_data = {}
        
        company_name = price_data.get("company_name", info.get("company_name", symbol)) or symbol
        
        # Direct query to Perplexity for all financial metrics
        prompt = f"""Search for complete financial data for {company_name} (ticker: {symbol.upper()}).

Return ONLY valid JSON with these exact fields:
{{
  "market_cap": number (in crores for Indian stocks),
  "enterprise_value": number,
  "pe_ratio": number,
  "forward_pe": number,
  "price_to_book": number,
  "price_to_sales": number,
  "ev_to_ebitda": number,
  "peg_ratio": number,
  "profit_margin": number (as percentage),
  "operating_margin": number,
  "gross_margin": number,
  "roe": number,
  "roa": number,
  "roic": number,
  "revenue_growth": number (YoY percentage),
  "earnings_growth": number,
  "eps": number,
  "dividend_yield": number,
  "debt_to_equity": number,
  "current_ratio": number,
  "quick_ratio": number,
  "beta": number,
  "fifty_two_week_high": number,
  "fifty_two_week_low": number,
  "sector": "string",
  "industry": "string"
}}

If exact value unavailable, provide your best estimate based on recent data. Use null for truly unavailable data. Output JSON only."""
        
        pplx_result = pplx._chat(
            messages=[
                {"role": "system", "content": "You are a financial data API. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        llm_metrics = pplx._parse_json(pplx_result.get("text", "{}"))
        
        # Build result with fallbacks from all sources
        result = {
            "symbol": symbol,
            "market_cap": llm_metrics.get("market_cap") or info.get("market_cap", 0) or price_data.get("market_cap", 0),
            "enterprise_value": llm_metrics.get("enterprise_value") or info.get("enterprise_value", 0),
            "pe_ratio": llm_metrics.get("pe_ratio") or info.get("pe_ratio", 0) or price_data.get("pe_ratio", 0),
            "forward_pe": llm_metrics.get("forward_pe") or info.get("forward_pe", 0),
            "price_to_book": llm_metrics.get("price_to_book") or info.get("price_to_book", 0),
            "price_to_sales": llm_metrics.get("price_to_sales") or info.get("price_to_sales", 0),
            "ev_to_ebitda": llm_metrics.get("ev_to_ebitda") or info.get("ev_to_ebitda", 0),
            "peg_ratio": llm_metrics.get("peg_ratio") or info.get("peg_ratio", 0),
            "eps": llm_metrics.get("eps") or info.get("eps", 0),
            "eps_growth": llm_metrics.get("eps_growth") or llm_metrics.get("earnings_growth") or info.get("eps_growth", 0),
            "revenue": llm_metrics.get("revenue") or info.get("revenue", 0),
            "revenue_growth": llm_metrics.get("revenue_growth") or info.get("revenue_growth", 0),
            "gross_margin": llm_metrics.get("gross_margin") or info.get("gross_margin", 0),
            "operating_margin": llm_metrics.get("operating_margin") or info.get("operating_margin", 0),
            "profit_margin": llm_metrics.get("profit_margin") or info.get("profit_margin", 0),
            "roe": llm_metrics.get("roe") or info.get("roe", 0),
            "roa": llm_metrics.get("roa") or info.get("roa", 0),
            "roic": llm_metrics.get("roic") or info.get("roic", 0),
            "dividend_yield": llm_metrics.get("dividend_yield") or info.get("dividend_yield", 0),
            "payout_ratio": llm_metrics.get("payout_ratio") or info.get("payout_ratio", 0),
            "current_ratio": llm_metrics.get("current_ratio") or info.get("current_ratio", 0),
            "quick_ratio": llm_metrics.get("quick_ratio") or info.get("quick_ratio", 0),
            "debt_to_equity": llm_metrics.get("debt_to_equity") or info.get("debt_to_equity", 0),
            "net_debt_to_ebitda": llm_metrics.get("net_debt_to_ebitda") or info.get("net_debt_to_ebitda", 0),
            "beta": llm_metrics.get("beta") or info.get("beta", 0) or price_data.get("beta", 0),
            "fifty_two_week_high": llm_metrics.get("fifty_two_week_high") or info.get("fifty_two_week_high", 0) or price_data.get("fifty_two_week_high", 0),
            "fifty_two_week_low": llm_metrics.get("fifty_two_week_low") or info.get("fifty_two_week_low", 0) or price_data.get("fifty_two_week_low", 0),
            "currency": price_data.get("currency", "USD"),
            "sector": llm_metrics.get("sector") or info.get("sector", price_data.get("sector", "Unknown")),
            "industry": llm_metrics.get("industry") or info.get("industry", price_data.get("industry", "Unknown")),
        }
        
        # Clean up None values to 0
        for key in result:
            if result[key] is None:
                result[key] = 0
        
        return json.dumps(result, indent=2)
    except Exception as e:
        import traceback
        print(f"[FINANCIAL METRICS] Error: {e}")
        print(traceback.format_exc())
        return json.dumps({"error": f"Failed to fetch financial metrics: {str(e)}"})


# TOOL-11: COMPARE-STOCKS
@tool
def compare_stocks(symbols: List[str]) -> str:
    """Compare multiple stocks using Perplexity quote data."""
    try:
        results = {}
        for symbol in symbols:
            try:
                price_data = _get_stock_price_data(symbol)

                results[symbol] = {
                    "price": price_data.get("current_price", 0),
                    "market_cap": price_data.get("market_cap", 0),
                    "pe_ratio": price_data.get("pe_ratio", 0),
                    "sector": price_data.get("sector", "Unknown"),
                    "industry": price_data.get("industry", "Unknown"),
                    "currency": price_data.get("currency", "USD"),
                }
            except Exception as e:
                results[symbol] = {"error": str(e)}
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# TOOL-12: CALCULATE-PORTFOLIO-METRICS
@tool
def calculate_portfolio_metrics(holdings: List[Dict]) -> str:
    """Calculate portfolio metrics using Perplexity quote data."""
    try:
        total_value = 0
        results = []
        for holding in holdings:
            symbol = holding.get("symbol")
            quantity = holding.get("quantity", 0)
            if symbol:
                price_data = _get_stock_price_data(symbol)
                current_price = price_data.get("current_price", 0)
                value = current_price * quantity
                total_value += value
                results.append(
                    {
                        "symbol": symbol,
                        "quantity": quantity,
                        "current_price": current_price,
                        "value": value,
                    }
                )
        return json.dumps(
            {
                "holdings": results,
                "total_value": total_value,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"Portfolio calculation failed: {str(e)}"})


# TOOL-13: ANALYZE-CHART
@tool
async def analyze_chart(symbol: str, period: str = "1mo") -> str:
    """Analyze stock chart patterns and technical signals using AI."""
    try:
        hist_data = _get_historical_data(symbol, period)
        if not hist_data or len(hist_data) < 5:
            return json.dumps({"error": "Insufficient historical data for chart analysis"})

        result = await backend.analyze_chart(symbol, hist_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Chart analysis failed: {str(e)}"})


# TOOL-14: SUMMARIZE-NEWS
@tool
async def summarize_news_articles(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
) -> str:
    """Get AI-powered summary of recent stock news."""
    try:
        articles = _get_stock_news_data(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )
        if not articles:
            return json.dumps({"summary": "No recent news articles found for this stock."})

        summaries = []
        for article in articles[:MAX_NEWS_ARTICLES_FOR_LLM]:
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            source = article.get("source", "Unknown")
            summaries.append(f"[{source}] {title}. {description}".strip())

        backend_result = await backend.summarize_news(summaries)
        return json.dumps(
            {
                "summary": backend_result.get("summary", ""),
                "article_count": backend_result.get("article_count", len(articles)),
                "sources": list(set(a.get("source", "Unknown") for a in articles[:MAX_NEWS_ARTICLES_FOR_LLM])),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"News summary failed: {str(e)}"})


# TOOL-15: ANALYZE-COMBINED-NEWS
@tool
async def analyze_combined_news(
    symbol: str,
    company_name: str,
    days: int = 7,
) -> str:
    """
    Analyze news from LiveMint + Google Search + NewsAPI and get AI-powered insights.
    Fetches top source-specific items and passes them to backend for analysis.
    """
    mint_articles: List[Dict] = []
    google_articles: List[Dict] = []
    newsapi_articles: List[Dict] = []

    try:
        mint_articles = mint_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            limit=10,
        )[:MAX_NEWS_ARTICLES_FOR_LLM]

        google_articles = _merge_news_articles(
            google_search_client.get_stock_news(
                symbol=symbol,
                company_name=company_name,
                days=days,
                limit=10,
            ),
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )

        newsapi_articles = _get_newsapi_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category="general",
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )

        print(
            f"[COMBINED NEWS] Mint: {len(mint_articles)} articles, "
            f"Google: {len(google_articles)} articles, "
            f"NewsAPI: {len(newsapi_articles)} articles"
        )

        if not mint_articles and not google_articles and not newsapi_articles:
            return json.dumps(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "analysis": "No recent news articles found from any source.",
                    "sentiment_summary": "neutral",
                    "articles_analyzed": 0,
                }
            )

        # Merge Mint + Google into the primary channel; pass NewsAPI as secondary.
        backend_primary_news = _merge_news_articles(
            [
                *[dict(a, source=a.get("source") or "LiveMint") for a in mint_articles],
                *[dict(a, source=a.get("source") or "Google Search") for a in google_articles],
            ],
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )
        result = await backend.analyze_combined_news(
            symbol=symbol,
            company_name=company_name,
            newsapi_articles=backend_primary_news,
            mint_articles=newsapi_articles,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        print(f"[COMBINED NEWS ERROR] {str(e)}")
        try:
            all_headlines = []
            sentiments = []

            for a in mint_articles[:MAX_NEWS_ARTICLES_FOR_LLM]:
                all_headlines.append(f"[Mint] {a.get('title', 'N/A')}")
                sentiments.append(a.get("sentiment", "neutral"))

            for a in google_articles[:MAX_NEWS_ARTICLES_FOR_LLM]:
                all_headlines.append(f"[Google] {a.get('title', 'N/A')}")
                sentiments.append(a.get("sentiment", "neutral"))

            for a in newsapi_articles[:MAX_NEWS_ARTICLES_FOR_LLM]:
                all_headlines.append(f"[NewsAPI] {a.get('title', 'N/A')}")
                sentiments.append(a.get("sentiment", "neutral"))

            positive = sentiments.count("positive")
            negative = sentiments.count("negative")

            return json.dumps(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "analysis": f"Backend analysis unavailable. Found {len(all_headlines)} articles.",
                    "sentiment_summary": "positive"
                    if positive > negative
                    else ("negative" if negative > positive else "neutral"),
                    "top_headlines": all_headlines,
                    "articles_analyzed": len(all_headlines),
                    "error": str(e),
                },
                indent=2,
            )
        except Exception:
            return json.dumps({"error": f"Combined news analysis failed: {str(e)}"})


# TOOL-16: SEARCH-GROUNDED-ANALYSIS
@tool
async def search_grounded_analysis(
    symbol: str,
    company_name: str,
    query_type: str = "analysis",
    time_frame: str = "3mo",
) -> str:
    """
    Get comprehensive grounded stock analysis using Perplexity.
    """
    try:
        news_days = _timeframe_to_days(time_frame)
        stock_data = _get_stock_price_data(symbol)
        news_articles = _get_stock_news_data(
            symbol=symbol,
            company_name=company_name,
            days=news_days,
            category="general",
            limit=MAX_NEWS_ARTICLES_FOR_LLM,
        )
        short_analysis = _build_short_stock_analysis(
            symbol=symbol,
            company_name=company_name,
            price_data=stock_data,
            news_articles=news_articles,
        )

        prompt = (
            f"Analyze {company_name} ({symbol}) for {query_type}. "
            f"Use latest available information over the {time_frame} horizon. "
            "Use this stock snapshot and recent headlines as grounding context, and provide "
            "a concise investment view with BUY/HOLD/SELL.\n\n"
            f"Stock Snapshot:\n{json.dumps(stock_data, indent=2)}\n\n"
            f"Recent News:\n{json.dumps(news_articles[:MAX_NEWS_ARTICLES_FOR_LLM], indent=2)}"
        )
        result = perplexity.quick_query(prompt)

        response = {
            "symbol": symbol,
            "company_name": company_name,
            "query_type": query_type,
            "time_frame": time_frame,
            "analysis": result.get("response", "No analysis available"),
            "short_analysis": short_analysis,
            "stock_data": stock_data,
            "news": news_articles,
            "news_count": len(news_articles),
            "sources": result.get("sources", []),
            "grounding_used": result.get("search_used", True),
            "model": result.get("model", "sonar"),
            "symbol_flow": {
                "requested_symbol": symbol,
                "alphavantage_symbol": alphavantage.normalize_symbol(symbol),
            },
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        print(f"[SEARCH ANALYSIS ERROR] {str(e)}")
        return json.dumps(
            {
                "error": f"Search-grounded analysis failed: {str(e)}",
                "symbol": symbol,
                "company_name": company_name,
                "fallback_suggestion": "Try using get_stock_news or analyze_combined_news instead",
            }
        )


# TOOL-17: QUICK-SEARCH-QUERY
@tool
async def quick_search_query(query: str) -> str:
    """
    Quick grounded search using Perplexity.
    """
    try:
        print(f"[QUICK SEARCH] Querying Perplexity: {query}")
        symbol, company_name, symbol_debug = _resolve_symbol_from_query(query)
        stock_data = {}
        news_articles: List[Dict] = []
        short_analysis = ""

        if symbol and company_name:
            try:
                stock_data = _get_stock_price_data(symbol)
            except Exception:
                stock_data = {}
            try:
                news_articles = _get_stock_news_data(
                    symbol=symbol,
                    company_name=company_name,
                    days=7,
                    category="general",
                    limit=MAX_NEWS_ARTICLES_FOR_LLM,
                )
            except Exception:
                news_articles = []

            short_analysis = _build_short_stock_analysis(
                symbol=symbol,
                company_name=company_name,
                price_data=stock_data,
                news_articles=news_articles,
            )

            enriched_query = (
                f"{query}\n\nUse this latest stock context in your answer:\n"
                f"Symbol: {symbol}\nCompany: {company_name}\n"
                f"Stock Snapshot: {json.dumps(stock_data, indent=2)}\n"
                f"Recent News: {json.dumps(news_articles[:MAX_NEWS_ARTICLES_FOR_LLM], indent=2)}"
            )
        else:
            enriched_query = query

        result = perplexity.quick_query(enriched_query)

        return json.dumps(
            {
                "query": query,
                "symbol": symbol,
                "company_name": company_name,
                "response": result.get("response", "No response"),
                "short_analysis": short_analysis,
                "stock_data": stock_data,
                "news": news_articles,
                "news_count": len(news_articles),
                "symbol_flow": symbol_debug,
                "sources": result.get("sources", []),
                "search_used": result.get("search_used", True),
                "model": result.get("model", "sonar"),
            },
            indent=2,
        )

    except Exception as e:
        print(f"[QUICK SEARCH ERROR] {str(e)}")
        return json.dumps(
            {
                "error": f"Quick search failed: {str(e)}",
                "query": query,
            }
        )


# TOOL-LIST
ALL_TOOLS = [
    get_stock_price,
    get_stock_info,
    get_hist_data,
    get_analyze_risk,
    predict_price,
    get_market_maker_quote,
    get_stock_news,
    analyze_news_sentiment,
    get_market_news,
    compare_stocks,
    get_financial_metrics,
    calculate_portfolio_metrics,
    analyze_chart,
    summarize_news_articles,
    analyze_combined_news,
    search_grounded_analysis,
    quick_search_query,
]
