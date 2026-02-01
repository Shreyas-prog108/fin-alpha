from langchain.tools import tool
from typing import List,Dict
import json

from .clients import (
    get_backend_client,
    get_tradingview_client,
    get_news_client,
    get_mint_client,
    get_yahoo_client
)

backend = get_backend_client()
tradingview = get_tradingview_client()
news_client = get_news_client()
mint_client = get_mint_client()
yahoo = get_yahoo_client()

#TOOL-1:STOCK-PRICE
@tool
def get_stock_price(symbol:str)->str:
    """Get current stock price using TradingView (fallback to Yahoo)"""
    try:
        # Try TradingView first (more reliable for NSE/BSE)
        print(f"[TOOLS] Fetching price for {symbol} from TradingView...")
        result = tradingview.get_current_price(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        # Fallback to Yahoo
        print(f"[TOOLS] TradingView failed: {str(e)[:100]}... Fallback to Yahoo")
        try:
            result = yahoo.get_current_price(symbol)
            return json.dumps(result, indent=2)
        except Exception as yahoo_error:
            return json.dumps({
                "error": f"TradingView failed: {str(e)}. Yahoo failed: {str(yahoo_error)}"
            })

#TOOL-2:STOCK-INFO
@tool
def get_stock_info(symbol:str)->str:
    """Get detailed stock info using Yahoo Finance"""
    try:
        result = yahoo.get_stock_info(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-3:HISTORICAL-DATA
@tool
def get_hist_data(symbol: str, period: str = "1mo") -> str:
    """Get comprehensive historical data of stock using Yahoo Finance"""
    try:
        result = yahoo.get_historical_data(symbol, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch historical data: {str(e)}"})
    
#TOOL-4:ANALYZE-RISK
@tool
async def get_analyze_risk(symbol:str, period:str="1mo")->str:
    """Get comprehensive analysis of risk using historical data"""
    try:
        hist_data = yahoo.get_historical_data(symbol, period)
        if not hist_data:
            return json.dumps({"error": "No historical data available"})
        
        # Call backend for risk analysis
        result = await backend.analyze_risk(symbol, hist_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Risk analysis failed: {str(e)}"})
    
#TOOL-5:PREDICT-PRICE
@tool
async def predict_price(symbol:str, method:str="ema", period:str="1mo")->str:
    """Predict stock price using historical data"""
    try:
        hist_data = yahoo.get_historical_data(symbol, period)
        if not hist_data:
            return json.dumps({"error": "No historical data available"})
        
        # Call backend for price prediction
        result = await backend.predict_price(symbol, hist_data, method=method)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Price prediction failed: {str(e)}"})
    
#TOOL-6:GET-MARKET-MAKER-QUOTE
@tool
async def get_market_maker_quote(symbol:str,risk_aversion:float = 0.1)->str:
    """Calculate optimal bid/ask using Avellaneda-Stoikov"""
    return json.dumps({
        "error": "Market maker quote requires volatility; provider removed"
    })
    
#TOOL-7:GET-STOCK-NEWS
@tool
def get_stock_news(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general"
) -> str:
    """Get comprehensive News of Stock"""
    try:
        articles = news_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category
        )
        return json.dumps({"articles": articles, "count": len(articles)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})

#TOOL-8:ANALYZE-SENTIMENT
@tool
def analyze_news_sentiment(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general"
) -> str:
    """Analyze the sentiment of market for company"""
    try:
        articles = news_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category
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
        
        return json.dumps({
            "overall_sentiment": overall,
            "average_score": round(avg_score, 2),
            "breakdown": {"positive": positive, "negative": negative, "neutral": neutral},
            "article_count": len(articles),
            "top_headlines": [a["title"] for a in articles[:5]]
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Sentiment analysis failed: {str(e)}"})

#TOOL-9:GET-MARKET-NEWS
@tool
def get_market_news(limit: int = 10) -> str:
    """Get market news"""
    try:
        articles = news_client.get_market_news(limit=limit)
        return json.dumps({"articles": articles, "count": len(articles)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch market news: {str(e)}"})

#TOOL-10:GET-FINANCIAL-METRICS
@tool
def get_financial_metrics(symbol:str)->str:
    """Get Financial Metrics"""
    return json.dumps({"error": "Financial metrics provider removed"})
    
#TOOL-11:COMPARE-STOCKS
@tool
def compare_stocks(symbols:List[str]) -> str:
    """Compare multiple stocks using Yahoo Finance"""
    try:
        results={}
        for symbol in symbols:
            try:
                price_data = yahoo.get_current_price(symbol)
                
                results[symbol]={
                    "price": price_data.get("current_price", 0),
                    "market_cap": price_data.get("market_cap", 0),
                    "pe_ratio": price_data.get("pe_ratio", 0),
                    "sector": price_data.get("sector", "Unknown"),
                    "industry": price_data.get("industry", "Unknown"),
                    "currency": price_data.get("currency", "INR")
                }
            except Exception as e:
                results[symbol]={"error": str(e)}
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-12:CALCULATE-PORTFOLIO-METRICS
@tool
def calculate_portfolio_metrics(holdings: List[Dict]) -> str:
    """Calculate portfolio metrics using Yahoo Finance data"""
    try:
        total_value = 0
        results = []
        for holding in holdings:
            symbol = holding.get("symbol")
            quantity = holding.get("quantity", 0)
            if symbol:
                price_data = yahoo.get_current_price(symbol)
                current_price = price_data.get("current_price", 0)
                value = current_price * quantity
                total_value += value
                results.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "current_price": current_price,
                    "value": value
                })
        return json.dumps({
            "holdings": results,
            "total_value": total_value
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Portfolio calculation failed: {str(e)}"})

#TOOL-13:ANALYZE-CHART
@tool
async def analyze_chart(symbol: str, period: str = "1mo") -> str:
    """Analyze stock chart patterns and technical signals using AI"""
    try:
        hist_data = yahoo.get_historical_data(symbol, period)
        if not hist_data or len(hist_data) < 5:
            return json.dumps({"error": "Insufficient historical data for chart analysis"})
        
        # Call backend for chart analysis
        result = await backend.analyze_chart(symbol, hist_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Chart analysis failed: {str(e)}"})

#TOOL-14:SUMMARIZE-NEWS
@tool
async def summarize_news_articles(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general"
) -> str:
    """Get AI-powered summary of recent news articles"""
    try:
        articles = news_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days,
            category=category
        )
        if not articles:
            return json.dumps({"summary": "No recent news articles found for this stock."})
        
        # Create a summary from the articles
        headlines = [a["title"] for a in articles[:10]]
        sentiments = [a.get("sentiment", "neutral") for a in articles]
        
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        
        return json.dumps({
            "article_count": len(articles),
            "sentiment_overview": f"{positive} positive, {negative} negative, {len(sentiments) - positive - negative} neutral",
            "recent_headlines": headlines,
            "sources": list(set(a.get("source", "Unknown") for a in articles[:10]))
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"News summary failed: {str(e)}"})


#TOOL-15:ANALYZE-COMBINED-NEWS
@tool
async def analyze_combined_news(
    symbol: str,
    company_name: str,
    days: int = 7
) -> str:
    """
    Analyze news from multiple sources (NewsAPI + LiveMint) and get AI-powered insights.
    Fetches 3 latest news from each source and passes to backend for comprehensive analysis.
    Returns sentiment analysis, key themes, and investment implications.
    """
    try:
        # Fetch news from NewsAPI
        newsapi_articles = news_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            days=days
        )[:3]  # Top 3 from NewsAPI
        
        # Fetch news from LiveMint
        mint_articles = mint_client.get_stock_news(
            symbol=symbol,
            company_name=company_name,
            limit=10
        )[:3]  
        
        print(f"[COMBINED NEWS] NewsAPI: {len(newsapi_articles)} articles, Mint: {len(mint_articles)} articles")
        
        if not newsapi_articles and not mint_articles:
            return json.dumps({
                "symbol": symbol,
                "company_name": company_name,
                "analysis": "No recent news articles found from any source.",
                "sentiment_summary": "neutral",
                "articles_analyzed": 0
            })
        
        # Call backend for AI analysis
        result = await backend.analyze_combined_news(
            symbol=symbol,
            company_name=company_name,
            newsapi_articles=newsapi_articles,
            mint_articles=mint_articles
        )
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"[COMBINED NEWS ERROR] {str(e)}")
        # Fallback: return basic aggregation without AI analysis
        try:
            all_headlines = []
            sentiments = []
            
            for a in newsapi_articles[:3]:
                all_headlines.append(f"[NewsAPI] {a.get('title', 'N/A')}")
                sentiments.append(a.get('sentiment', 'neutral'))
            
            for a in mint_articles[:3]:
                all_headlines.append(f"[Mint] {a.get('title', 'N/A')}")
                sentiments.append(a.get('sentiment', 'neutral'))
            
            positive = sentiments.count("positive")
            negative = sentiments.count("negative")
            
            return json.dumps({
                "symbol": symbol,
                "company_name": company_name,
                "analysis": f"Backend analysis unavailable. Found {len(all_headlines)} articles.",
                "sentiment_summary": "positive" if positive > negative else ("negative" if negative > positive else "neutral"),
                "top_headlines": all_headlines,
                "articles_analyzed": len(all_headlines),
                "error": str(e)
            }, indent=2)
        except:
            return json.dumps({"error": f"Combined news analysis failed: {str(e)}"})


#TOOL-16:SEARCH-GROUNDED-ANALYSIS
@tool
async def search_grounded_analysis(
    symbol: str,
    company_name: str,
    query_type: str = "analysis",
    time_frame: str = "3mo"
) -> str:
    """
    Get comprehensive stock analysis using Groq.
    This tool provides AI-powered investment recommendations.
    
    Args:
        symbol: Stock ticker (e.g., 'RELIANCE.NSE', 'AAPL')
        company_name: Company name (e.g., 'Reliance Industries', 'Apple Inc')
        query_type: Type of analysis - 'analysis', 'news', 'sentiment', 'recommendation'
        time_frame: Time frame for analysis - '1wk', '1mo', '3mo', '6mo', '1y'
    
    Returns:
        Comprehensive analysis with real-time data, sources, and recommendations.
    """
    try:
        print(f"[SEARCH ANALYSIS] Querying Groq for {symbol} ({company_name})")
        
        result = await backend.search_analysis(
            symbol=symbol,
            company_name=company_name,
            query_type=query_type,
            time_frame=time_frame
        )
        
        # Format the response
        response = {
            "symbol": symbol,
            "company_name": company_name,
            "query_type": query_type,
            "time_frame": time_frame,
            "analysis": result.get("analysis", "No analysis available"),
            "sources": result.get("sources", []),
            "grounding_used": result.get("grounding_used", False),
            "model": result.get("model", "openai/gpt-oss-120b")
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        print(f"[SEARCH ANALYSIS ERROR] {str(e)}")
        return json.dumps({
            "error": f"Search-grounded analysis failed: {str(e)}",
            "symbol": symbol,
            "company_name": company_name,
            "fallback_suggestion": "Try using get_stock_news or analyze_combined_news instead"
        })


#TOOL-17:QUICK-SEARCH-QUERY
@tool
async def quick_search_query(query: str) -> str:
    """
    Quick search using Groq.
    Use this for general market queries, news lookups, or quick information retrieval.
    
    Args:
        query: Natural language query (e.g., 'What happened to TCS stock today?')
    
    Returns:
        AI-generated response with web-sourced information.
    """
    try:
        print(f"[QUICK SEARCH] Querying: {query}")
        
        result = await backend.query_groq(prompt=query)
        
        return json.dumps({
            "query": query,
            "response": result.get("response", "No response"),
            "sources": result.get("sources", []),
            "search_used": result.get("search_used", False)
        }, indent=2)
        
    except Exception as e:
        print(f"[QUICK SEARCH ERROR] {str(e)}")
        return json.dumps({
            "error": f"Quick search failed: {str(e)}",
            "query": query
        })

    
#TOOL-LIST
ALL_TOOLS=[
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
    quick_search_query
]
    