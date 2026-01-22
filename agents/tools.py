from langchain.tools import tool
from typing import List,Dict
import json

from .clients import (
    get_backend_client,
    get_tradingview_client
)

backend = get_backend_client()
tradingview = get_tradingview_client()

#TOOL-1:STOCK-PRICE
@tool
def get_stock_price(symbol:str)->str:
    """Get current stock price and stock info using TradingView"""
    try:
        result = tradingview.get_current_price(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": f"TradingView failed: {str(e)}"})

#TOOL-2:STOCK-INFO
@tool
def get_stock_info(symbol:str)->str:
    """Get stock info using TradingView"""
    try:
        result = tradingview.get_current_price(symbol)
        stock_info = {
            "symbol": result.get("symbol", symbol),
            "company_name": result.get("company_name", ""),
            "description": result.get("description", ""),
            "sector": result.get("sector", ""),
            "market_cap": result.get("market_cap", 0),
            "pe_ratio": result.get("pe_ratio", 0),
            "currency": result.get("currency", ""),
            "exchange": result.get("exchange", "")
        }
        return json.dumps(stock_info,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-3:HISTORICAL-DATA
@tool
def get_hist_data(symbol: str, period: str = "1mo") -> str:
    """Get comprehensive historical data of stock"""
    return json.dumps({
        "error": "Historical data provider not configured"
    })
    
#TOOL-4:ANALYZE-RISK
@tool
async def get_analyze_risk(symbol:str,period:str="1mo")->str:
    """Get comprehensive analysis of risk"""
    return json.dumps({
        "error": "Risk analysis requires historical data; provider removed"
    })
    
#TOOL-5:PREDICT-PRICE
@tool
async def predict_price(symbol:str,method:str="ema",period:str="1mo")->str:
    """Predict stock price"""
    return json.dumps({
        "error": "Price prediction requires historical data; provider removed"
    })
    
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
    return json.dumps({"error": "News API removed"})

#TOOL-8:ANALYZE-SENTIMENT
@tool
def analyze_news_sentiment(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general"
) -> str:
    """Analyze the sentiment of market for company"""
    return json.dumps({"error": "News API removed"})

#TOOL-9:GET-MARKET-NEWS
@tool
def get_market_news(limit: int = 10) -> str:
    """Get market news"""
    return json.dumps({"error": "News API removed"})

#TOOL-10:GET-FINANCIAL-METRICS
@tool
def get_financial_metrics(symbol:str)->str:
    """Get Financial Metrics"""
    return json.dumps({"error": "Financial metrics provider removed"})
    
#TOOL-11:COMPARE-STOCKS
@tool
def compare_stocks(symbols:List[str]) -> str:
    """Compare multiple stocks using TradingView"""
    try:
        results={}
        for symbol in symbols:
            try:
                price_data=tradingview.get_current_price(symbol)
                
                results[symbol]={
                    "price": price_data["current_price"],
                    "market_cap": price_data["market_cap"],
                    "pe_ratio": price_data["pe_ratio"],
                    "volatility": None,
                    "sector": price_data.get("sector", "Unknown"),
                    "change_percent": price_data.get("change_percent", 0)
                }
            except Exception as e:
                results[symbol]={"error": str(e)}
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-12:CALCULATE-PORTFOLIO-METRICS
@tool
def calculate_portfolio_metrics(holdings: List[Dict]) -> str:
    """Calculate portfolio metrics"""
    return json.dumps({
        "error": "Portfolio metrics require volatility data; provider removed"
    })

#TOOL-13:ANALYZE-CHART
@tool
async def analyze_chart(symbol: str, period: str = "1mo") -> str:
    """Analyze stock chart patterns and technical signals using AI"""
    return json.dumps({
        "error": "Chart analysis requires historical data; provider removed"
    })

#TOOL-14:SUMMARIZE-NEWS
@tool
async def summarize_news_articles(
    symbol: str,
    company_name: str,
    days: int = 7,
    category: str = "general"
) -> str:
    """Get AI-powered summary of recent news articles"""
    return json.dumps({"error": "News API removed"})
    
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
    summarize_news_articles
]
    