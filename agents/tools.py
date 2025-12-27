from langchain.tools import tool
from typing import List,Dict
import json

from .clients import (
    get_yahoo_client, 
    get_news_client, 
    get_backend_client
)

from .prompts.tools_prompts import TOOL_DESCRIPTIONS

yahoo=get_yahoo_client()
news=get_news_client()
backend=get_backend_client()

#TOOL-1:STOCK-PRICE
@tool(description=TOOL_DESCRIPTIONS["get_stock_price"])
def get_stock_price(symbol:str)->str:
    """Get current stock price and stock info"""
    try:
        result=yahoo.get_current_price(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-2:STOCK-INFO
@tool(description=TOOL_DESCRIPTIONS["get_stock_info"])
def get_stock_info(symbol:str)->str:
    """Get comprehensive stock information"""
    try:
        result=yahoo.get_stock_info(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-3:HISTORICAL-DATA
@tool(description=TOOL_DESCRIPTIONS["get_hist_data"])
def get_hist_data(symbol:str,period:str="1mo")->str:
    """Get comprehensive historical data of stock"""
    try:
        result=yahoo.get_hist_data(symbol,period)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-4:ANALYZE-RISK
@tool(description=TOOL_DESCRIPTIONS["get_analyze_risk"])
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
@tool(description=TOOL_DESCRIPTIONS["predict_stock_price"])
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
@tool(description=TOOL_DESCRIPTIONS["get_market_maker_quote"])
async def get_market_maker_quote(symbol:str,risk_aversion:float = 0.1)->str:
    """Calculate optimal bid/ask using Avellaneda-Stoikov"""
    try:
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
@tool(description=TOOL_DESCRIPTIONS["get_stock_news"])
def get_stock_news(symbol:str,company_name:str)->str:
    """Get comprehensive News of Stock"""
    try:
        result=news.get_news(company_name, days=7)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-8:ANALYZE-SENTIMENT
@tool(description=TOOL_DESCRIPTIONS["analyze_news_sentiment"])
def analyze_news_sentiment(symbol:str,company_name:str)->str:
    """analyze the sentiment of market for company"""
    try:
        result=news.analyze_sentiment(company_name)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-9:GET-MARKET-NEWS
@tool(description=TOOL_DESCRIPTIONS["get_market_news"])
def get_market_news(limit:int=10)->str:
    """Get market news"""
    try:
        result=yahoo.get_market_news(limit)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

#TOOL-10:GET-FINANCIAL-METRICS
@tool(description=TOOL_DESCRIPTIONS["get_financial_metrics"])
def get_financial_metrics(symbol:str)->str:
    """Get Financial Metrics"""
    try:
        result=yahoo.get_financial_metrics(symbol)
        return json.dumps(result,indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-11:COMPARE-STOCKS
@tool(description=TOOL_DESCRIPTIONS["compare_stocks"])
def compare_stocks(symbols:List[str]) -> str:
    """Compare multiple stocks"""
    try:
        results={}
        for symbol in symbols:
            price_data=yahoo.get_current_price(symbol)
            volatility=yahoo.calculate_volatility(symbol)
            
            results[symbol]={
                "price": price_data["current_price"],
                "market_cap": price_data["market_cap"],
                "pe_ratio": price_data["pe_ratio"],
                "volatility": volatility,
                "sector": price_data["sector"]
            }
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
#TOOL-12:CALCULATE-PORTFOLIO-METRICS
@tool(description=TOOL_DESCRIPTIONS["calculate_portfolio_metrics"])
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
    calculate_portfolio_metrics
]
    