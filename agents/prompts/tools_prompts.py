"""
Tool Descriptions and Usage Guidance for FinTerm Agent
Compact version with essential information for all tools
"""

from typing import Dict

# ========== TOOL DESCRIPTIONS ==========

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "get_stock_price": "Get current price, market cap, P/E ratio. Use for: price queries, basic info.",
    "get_stock_info": "Get detailed financials, ratios, company info. Use for: fundamental analysis.",
    "get_hist_data": "Fetch price/volume history. Use for: trend analysis, charts.",
    "get_financial_metrics": "Get profitability, valuation, health metrics. Use for: financial health check.",
    "get_analyze_risk": "Calculate volatility, risk level, anomalies. Use for: risk assessment. Returns: volatility, risk_level (low/medium/high).",
    "predict_stock_price": "Predict next price using EMA or linear regression. Use for: price forecasts. Returns: predicted_price, change_percent, confidence.",
    "get_market_maker_quote": "Calculate optimal bid/ask using Avellaneda-Stoikov. Use for: market making, fair pricing.",
    "get_stock_news": "Fetch recent news articles. Use for: news context, before sentiment.",
    "analyze_news_sentiment": "Analyze sentiment from news. Use for: market sentiment check. Returns: overall_sentiment, sentiment_score.",
    "get_market_news": "Get general market news. Use for: market conditions, macro trends.",
    "compare_stocks": "Compare multiple stocks side-by-side. Use for: stock comparison, relative analysis.",
    "calculate_portfolio_metrics": "Calculate portfolio volatility, return, Sharpe ratio. Use for: portfolio analysis.",
    "analyze_combined_news": "Analyze news from NewsAPI + LiveMint with AI. Use for: comprehensive news analysis.",
    "search_grounded_analysis": "🔥 PREFERRED: Get comprehensive analysis using Gemini + Google Search. Real-time news, prices, sentiment. Use for: investment decisions, news, recommendations.",
    "quick_search_query": "Quick real-time search for any market query. Use for: general questions, quick lookups.",
}

# ========== TOOL SELECTION GUIDE ==========

TOOL_SELECTION_GUIDE = """
Query Type → Tools:

⚡ PREFERRED: Use search_grounded_analysis for most queries - it provides real-time data!

1. Current Price: get_stock_price OR search_grounded_analysis (more reliable)
2. Risk Check: get_analyze_risk + get_stock_info
3. Price Prediction: predict_price + get_hist_data
4. Investment Decision (comprehensive):
   - search_grounded_analysis (BEST - gets everything via Google Search)
   - OR fallback to: get_stock_price + get_analyze_risk + analyze_news_sentiment + predict_price
5. News & Sentiment: search_grounded_analysis OR analyze_combined_news
6. Comparison: compare_stocks
7. Market Overview: quick_search_query("market news today")
8. Portfolio: calculate_portfolio_metrics
9. General Questions: quick_search_query

General Rule: PREFER search_grounded_analysis for any investment/analysis query.
Fallback to other tools if search fails or for specific data needs.
"""

# ========== USAGE EXAMPLES ==========

USAGE_EXAMPLES = """
Q: "What's AAPL price?" → search_grounded_analysis("AAPL", "Apple Inc", "analysis") OR get_stock_price("AAPL")
Q: "Is TSLA risky?" → search_grounded_analysis("TSLA", "Tesla", "analysis")
Q: "Should I buy NVDA?" → search_grounded_analysis("NVDA", "NVIDIA", "recommendation")
Q: "What's happening in the market?" → quick_search_query("Indian stock market news today")
Q: "Compare AAPL vs GOOGL" → compare_stocks(["AAPL", "GOOGL"])
Q: "Reliance news" → search_grounded_analysis("RELIANCE.NS", "Reliance Industries", "news")
"""

# ========== TOOL CATEGORIES ==========

TOOL_CATEGORIES = {
    "search_powered": ["search_grounded_analysis", "quick_search_query"],  # PREFER THESE
    "market_data": ["get_stock_price", "get_stock_info", "get_hist_data"],
    "risk_analysis": ["get_analyze_risk", "get_market_maker_quote"],
    "prediction": ["predict_price"],
    "sentiment": ["get_stock_news", "analyze_news_sentiment", "get_market_news", "analyze_combined_news"],
    "comparison": ["compare_stocks", "calculate_portfolio_metrics"]
}

# ========== DECISION TREE ==========

DECISION_TREE = """
1. Specific stock? → Extract symbol + company name
2. Query type?
   - Investment/Analysis → search_grounded_analysis (BEST)
   - Price Only → get_stock_price OR search_grounded_analysis
   - Risk → get_analyze_risk
   - Prediction → predict_price
   - News → search_grounded_analysis(type="news") OR analyze_combined_news
   - Sentiment → search_grounded_analysis(type="sentiment")
   - Compare → compare_stocks
   - General → quick_search_query
3. Synthesize & respond

⚡ DEFAULT TO search_grounded_analysis - it has real-time Google Search!
"""

__all__ = ['TOOL_DESCRIPTIONS', 'TOOL_SELECTION_GUIDE', 'USAGE_EXAMPLES', 'TOOL_CATEGORIES', 'DECISION_TREE']