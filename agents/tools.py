from langchain.tools import tool
from typing import List,Dict
import json

from .clients import (
    get_yahoo_client, 
    get_news_client, 
    get_backend_client,
    get_tradingview_client
)

yahoo=get_yahoo_client()
news=get_news_client()
backend=get_backend_client()
tradingview=get_tradingview_client()

#TOOL-1:STOCK-PRICE
@tool
def get_stock_price(symbol:str)->str:
    """Get current stock price and stock info using TradingView (primary) or Yahoo Finance (fallback)"""
    try:
        result=tradingview.get_current_price(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        print(f"[TRADINGVIEW FAILED] Falling back to Yahoo Finance: {str(e)}")
        try:
            result=yahoo.get_current_price(symbol)
            return json.dumps(result,indent=2)
        except Exception as e2:
            return json.dumps({"error": f"Both TradingView and Yahoo failed: {str(e2)})"})

#TOOL-2:STOCK-INFO
@tool
def get_stock_info(symbol:str)->str:
    """Get comprehensive stock information"""
    try:
        result=yahoo.get_stock_info(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-3:HISTORICAL-DATA
@tool
def get_hist_data(symbol:str,period:str="1mo")->str:
    """Get comprehensive historical data of stock"""
    try:
        result=yahoo.get_hist_data(symbol,period)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-4:ANALYZE-RISK
@tool
async def get_analyze_risk(symbol:str,period:str="1mo")->str:
    """Get comprehensive analysis of risk"""
    try:
        hist_data=yahoo.get_historical_data(symbol,period)
        data_for_backend=[{
                "time": d["date"],
                "close": d["close"],
                "volume": d["volume"]
            }
            for d in hist_data
        ]
        result=await backend.get_analyze_risk(symbol,data_for_backend)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-5:PREDICT-PRICE
@tool
async def predict_price(symbol:str,method:str="ema",period:str="1mo")->str:
    """Predict stock price"""
    try:
        hist_data=yahoo.get_historical_data(symbol, period)
        data_for_backend=[{"close": d["close"]} for d in hist_data]
        result=await backend.predict_price(symbol,data_for_backend,method)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-6:GET-MARKET-MAKER-QUOTE
@tool
async def get_market_maker_quote(symbol:str,risk_aversion:float = 0.1)->str:
    """Calculate optimal bid/ask using Avellaneda-Stoikov"""
    try:
        try:
            price_data=tradingview.get_current_price(symbol)
        except:
            price_data=yahoo.get_current_price(symbol)
        
        volatility=yahoo.calculate_volatility(symbol)
        result = await backend.get_market_maker_quote(
            mid_price=price_data["current_price"],
            volatility=volatility,
            risk_aversion=risk_aversion
        )
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-7:GET-STOCK-NEWS
@tool
def get_stock_news(symbol:str,company_name:str)->str:
    """Get comprehensive News of Stock"""
    try:
        result=news.get_stock_news(symbol, company_name, days=7)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-8:ANALYZE-SENTIMENT
@tool
def analyze_news_sentiment(symbol:str,company_name:str)->str:
    """analyze the sentiment of market for company"""
    try:
        result=news.analyze_news_sentiment(symbol, company_name)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-9:GET-MARKET-NEWS
@tool
def get_market_news(limit:int=10)->str:
    """Get market news"""
    try:
        result=yahoo.get_market_news(limit)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-10:GET-FINANCIAL-METRICS
@tool
def get_financial_metrics(symbol:str)->str:
    """Get Financial Metrics"""
    try:
        result=yahoo.get_financial_metrics(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-11:COMPARE-STOCKS
@tool
def compare_stocks(symbols:List[str]) -> str:
    """Compare multiple stocks using TradingView"""
    try:
        results={}
        for symbol in symbols:
            try:
                price_data=tradingview.get_current_price(symbol)
                volatility=yahoo.calculate_volatility(symbol)
                
                results[symbol]={
                    "price": price_data["current_price"],
                    "market_cap": price_data["market_cap"],
                    "pe_ratio": price_data["pe_ratio"],
                    "volatility": volatility,
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
    try:
        total_weight = sum(h["weight"] for h in holdings)
        if abs(total_weight - 1.0) > 0.01:
            return json.dumps({
                "error": f"Weights must sum to 1.0, got {total_weight}"
            })
        portfolio_volatility = 0.0
        portfolio_value = 0.0
        for holding in holdings:
            symbol = holding["symbol"]
            weight = holding["weight"]
            vol = yahoo.calculate_volatility(symbol)
            price = yahoo.get_current_price(symbol)
            portfolio_volatility += (weight ** 2) * (vol ** 2)
        portfolio_volatility = portfolio_volatility ** 0.5
        result = {
            "portfolio_volatility": portfolio_volatility,
            "risk_level": "low" if portfolio_volatility < 0.2 else "medium" if portfolio_volatility < 0.4 else "high",
            "holdings": holdings
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-13:ANALYZE-CHART
@tool
async def analyze_chart(symbol:str)->str:
    """Analyze stock chart patterns and technical signals using AI"""
    try:
        hist_data=yahoo.get_historical_data(symbol,"1mo")
        data_for_backend=[{
                "time": d["date"],
                "open": d["open"],
                "high": d["high"],
                "low": d["low"],
                "close": d["close"],
                "volume": d["volume"]
            }
            for d in hist_data
        ]
        result=await backend.analyze_chart(symbol,data_for_backend)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-14:SUMMARIZE-NEWS
@tool
async def summarize_news_articles(symbol:str,company_name:str)->str:
    """Get AI-powered summary of recent news articles"""
    try:
        articles_data=news.get_stock_news(symbol, company_name, days=7)
        if not articles_data:
            return json.dumps({"summary": "No recent news available"})
        article_texts = []
        for article in articles_data[:10]: 
            text = f"{article.get('title', '')}\n{article.get('description', '')}"
            article_texts.append(text)
        
        result=await backend.summarize_news(article_texts)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
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
    