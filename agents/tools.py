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
)

backend = get_backend_client()
alphavantage = get_alphavantage_client()
mint_client = get_mint_client()
perplexity = get_perplexity_client()
news_api_client = get_news_client()

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
    """Fetch quote using Perplexity first, then Alpha Vantage fallback."""
    try:
        requested_symbol = (symbol or "").strip().upper()
        normalized_symbol = _canonical_symbol(requested_symbol)
        result = perplexity.get_stock_price(symbol)
        returned_symbol = result.get("symbol", requested_symbol)
        if not _symbols_equivalent(normalized_symbol, returned_symbol):
            raise Exception(
                "Perplexity symbol mismatch "
                f"(requested={normalized_symbol}, returned={returned_symbol})"
            )
        result["requested_symbol"] = requested_symbol
        result["normalized_symbol"] = normalized_symbol
        return result
    except Exception as perplexity_error:
        try:
            requested_symbol = (symbol or "").strip().upper()
            normalized_symbol = _canonical_symbol(requested_symbol)
            fallback = alphavantage.get_current_price(normalized_symbol)
            fallback["source"] = "alphavantage_fallback"
            fallback["requested_symbol"] = requested_symbol
            fallback["alphavantage_symbol"] = normalized_symbol
            fallback["warning"] = f"Perplexity failed: {str(perplexity_error)}"
            return fallback
        except Exception as alpha_error:
            raise Exception(
                f"Perplexity quote failed: {str(perplexity_error)}. "
                f"Alpha Vantage fallback failed: {str(alpha_error)}"
            )


def _get_stock_info_data(symbol: str) -> Dict:
    """Fetch detailed info from Perplexity with symbol consistency checks."""
    requested_symbol = (symbol or "").strip().upper()
    normalized_symbol = _canonical_symbol(requested_symbol)
    try:
        result = perplexity.get_stock_info(symbol)
        returned_symbol = result.get("symbol", requested_symbol)
        if not _symbols_equivalent(normalized_symbol, returned_symbol):
            raise Exception(
                "Perplexity symbol mismatch "
                f"(requested={normalized_symbol}, returned={returned_symbol})"
            )
        result["requested_symbol"] = requested_symbol
        result["normalized_symbol"] = normalized_symbol
        return result
    except Exception as perplexity_error:
        fallback_quote = alphavantage.get_current_price(normalized_symbol)
        return {
            "symbol": normalized_symbol,
            "company_name": fallback_quote.get("company_name", normalized_symbol),
            "exchange": fallback_quote.get("exchange", ""),
            "currency": fallback_quote.get("currency", ""),
            "sector": "",
            "industry": "",
            "country": "",
            "market_cap": fallback_quote.get("market_cap", 0),
            "enterprise_value": 0,
            "pe_ratio": 0,
            "forward_pe": 0,
            "price_to_book": 0,
            "eps": 0,
            "dividend_yield": 0,
            "beta": 0,
            "fifty_two_week_high": 0,
            "fifty_two_week_low": 0,
            "description": "",
            "source": "alphavantage_fallback",
            "requested_symbol": requested_symbol,
            "alphavantage_symbol": normalized_symbol,
            "warning": f"Perplexity failed: {str(perplexity_error)}",
        }


def _get_stock_news_data(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general",
    limit: int = 10,
) -> List[Dict]:
    """Fetch stock news from Perplexity + NewsAPI + LiveMint."""
    perplexity_articles: List[Dict] = []
    newsapi_articles: List[Dict] = []
    mint_articles: List[Dict] = []
    errors = []

    try:
        perplexity_articles = perplexity.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
            limit=limit,
        )
    except Exception as e:
        errors.append(f"Perplexity: {str(e)}")

    try:
        mint_articles = mint_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            limit=limit,
        )
    except Exception as e:
        errors.append(f"LiveMint: {str(e)}")

    try:
        newsapi_articles = news_api_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category,
        )[: limit * 2]
    except Exception as e:
        errors.append(f"NewsAPI: {str(e)}")

    perplexity_articles = _filter_relevant_articles(perplexity_articles, symbol, company_name)
    newsapi_articles = _filter_relevant_articles(newsapi_articles, symbol, company_name)
    mint_articles = _filter_relevant_articles(mint_articles, symbol, company_name)

    merged = _merge_news_articles(
        [
            *perplexity_articles,
            *newsapi_articles,
            *mint_articles,
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
            limit=20,
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
            limit=20,
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
                "top_headlines": [a.get("title", "") for a in articles[:5]],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"Sentiment analysis failed: {str(e)}"})


# TOOL-9: GET-MARKET-NEWS
@tool
def get_market_news(limit: int = 10) -> str:
    """Get broad market news using Perplexity + NewsAPI."""
    try:
        perplexity_articles = perplexity.get_market_news(limit=limit)
        newsapi_articles = news_api_client.get_market_news(limit=limit)
        articles = _merge_news_articles(perplexity_articles + newsapi_articles, limit=limit)
        return json.dumps({"articles": articles, "count": len(articles)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch market news: {str(e)}"})


# TOOL-10: GET-FINANCIAL-METRICS
@tool
def get_financial_metrics(symbol: str) -> str:
    """Get Financial Metrics."""
    return json.dumps({"error": "Financial metrics provider removed"})


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
            limit=20,
        )
        if not articles:
            return json.dumps({"summary": "No recent news articles found for this stock."})

        summaries = []
        for article in articles[:20]:
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            source = article.get("source", "Unknown")
            summaries.append(f"[{source}] {title}. {description}".strip())

        backend_result = await backend.summarize_news(summaries)
        return json.dumps(
            {
                "summary": backend_result.get("summary", ""),
                "article_count": backend_result.get("article_count", len(articles)),
                "sources": list(set(a.get("source", "Unknown") for a in articles[:10])),
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
    Analyze news from Perplexity + NewsAPI + LiveMint and get AI-powered insights.
    Fetches top source-specific items and passes them to backend for analysis.
    """
    perplexity_articles: List[Dict] = []
    newsapi_articles: List[Dict] = []
    mint_articles: List[Dict] = []

    try:
        perplexity_articles = _merge_news_articles(perplexity.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category="general",
            limit=10,
        ), limit=3)

        newsapi_articles = _get_newsapi_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category="general",
            limit=3,
        )

        mint_articles = mint_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            limit=10,
        )[:3]

        print(
            f"[COMBINED NEWS] Perplexity: {len(perplexity_articles)} articles, "
            f"NewsAPI: {len(newsapi_articles)} articles, "
            f"Mint: {len(mint_articles)} articles"
        )

        if not perplexity_articles and not newsapi_articles and not mint_articles:
            return json.dumps(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "analysis": "No recent news articles found from any source.",
                    "sentiment_summary": "neutral",
                    "articles_analyzed": 0,
                }
            )

        # Keep backend payload key names for compatibility.
        # Merge Perplexity + NewsAPI into the `newsapi_articles` channel.
        backend_primary_news = _merge_news_articles(
            [
                *[dict(a, source=a.get("source") or "Perplexity") for a in perplexity_articles],
                *[dict(a, source=a.get("source") or "NewsAPI") for a in newsapi_articles],
            ],
            limit=6,
        )
        result = await backend.analyze_combined_news(
            symbol=symbol,
            company_name=company_name,
            newsapi_articles=backend_primary_news,
            mint_articles=mint_articles,
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        print(f"[COMBINED NEWS ERROR] {str(e)}")
        try:
            all_headlines = []
            sentiments = []

            for a in perplexity_articles[:3]:
                all_headlines.append(f"[Perplexity] {a.get('title', 'N/A')}")
                sentiments.append(a.get("sentiment", "neutral"))

            for a in newsapi_articles[:3]:
                all_headlines.append(f"[NewsAPI] {a.get('title', 'N/A')}")
                sentiments.append(a.get("sentiment", "neutral"))

            for a in mint_articles[:3]:
                all_headlines.append(f"[Mint] {a.get('title', 'N/A')}")
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
            limit=8,
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
            f"Recent News:\n{json.dumps(news_articles[:5], indent=2)}"
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
                    limit=5,
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
                f"Recent News: {json.dumps(news_articles, indent=2)}"
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
