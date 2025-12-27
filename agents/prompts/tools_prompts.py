"""
Tool Descriptions and Usage Guidance for FinTerm Agent
Compact version with essential information for all 12 tools
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
}

# ========== TOOL SELECTION GUIDE ==========

TOOL_SELECTION_GUIDE = """
Query Type → Tools:

1. Current Price: get_stock_price
2. Risk Check: analyze_stock_risk + get_stock_info
3. Price Prediction: predict_stock_price + get_historical_data
4. Investment Decision (comprehensive):
   - get_stock_price
   - analyze_stock_risk
   - analyze_news_sentiment
   - predict_stock_price
   - get_financial_metrics
5. Comparison: compare_stocks
6. Sentiment: analyze_news_sentiment + get_stock_news
7. Market Overview: get_market_news
8. Portfolio: calculate_portfolio_metrics

General Rule: Start simple, add tools as needed. Always include risk for investment decisions.
"""

# ========== USAGE EXAMPLES ==========

USAGE_EXAMPLES = """
Q: "What's AAPL price?" → get_stock_price("AAPL")
Q: "Is TSLA risky?" → analyze_stock_risk("TSLA")
Q: "Should I buy NVDA?" → [5 tools: price, risk, sentiment, prediction, metrics]
Q: "Compare AAPL vs GOOGL" → compare_stocks(["AAPL", "GOOGL"])
"""

# ========== TOOL CATEGORIES ==========

TOOL_CATEGORIES = {
    "market_data": ["get_stock_price", "get_stock_info", "get_historical_data", "get_financial_metrics"],
    "risk_analysis": ["analyze_stock_risk", "get_market_maker_quote"],
    "prediction": ["predict_stock_price"],
    "sentiment": ["get_stock_news", "analyze_news_sentiment", "get_market_news"],
    "comparison": ["compare_stocks", "calculate_portfolio_metrics"]
}

# ========== DECISION TREE ==========

DECISION_TREE = """
1. Specific stock? → Extract symbol
2. Query type?
   - Price → get_stock_price
   - Risk → analyze_stock_risk
   - Investment → ALL tools
   - Prediction → predict_stock_price
   - Sentiment → analyze_news_sentiment
   - Compare → compare_stocks
3. Synthesize & respond
"""

__all__ = ['TOOL_DESCRIPTIONS', 'TOOL_SELECTION_GUIDE', 'USAGE_EXAMPLES', 'TOOL_CATEGORIES', 'DECISION_TREE']