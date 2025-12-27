from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Tuple, Optional
import os
import json
import requests

from .state import AgentState
from .config import config
from .tools import ALL_TOOLS
from .prompts import (
    MAIN_AGENT_SYSTEM_PROMPT,
    TOOL_SELECTION_GUIDE,
    SYNTHESIS_PROMPT,
)

class FinAgent:
    """
    Main Financial Analysis Agent using LangGraph
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        google_api_key=config.GEMINI_API_KEY
        )
        self.graph=self._build_graph()
        self.ticker_cache = {}
        
        # Common abbreviations to full ticker mapping
        self.abbreviation_map = {
            "SBI": "SBIN.NS",
            "HDFC": "HDFCBANK.NS",
            "HDFC BANK": "HDFCBANK.NS",
            "ICICI": "ICICIBANK.NS",
            "ICICI BANK": "ICICIBANK.NS",
            "IDBI": "IDBI.NS",
            "PNB": "PNB.NS",
            "BOB": "BANKBARODA.NS",
            "BOI": "BANKINDIA.NS",
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "RELIANCE": "RELIANCE.NS",
        }

    def _search_tradingview(self, company_name: str) -> Optional[str]:
        """
        Search using TradingView API (Primary)
        Better for Indian stocks and global coverage
        """
        try:
            url = os.getenv("TRADINGVIEW_API_URL","https://symbol-search.tradingview.com/symbol_search/")
            params = {
                'text': company_name,
                'type': 'stock',
                'exchange': '',
                'lang': 'en'
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data and len(data) > 0:
                best_match = data[0]
                symbol = best_match.get('symbol', '')
                description = best_match.get('description', '')
                exchange = best_match.get('exchange', '')
                
                ticker = symbol.split(':')[-1] if ':' in symbol else symbol
                
                if exchange in ['NSE', 'NSI']:
                    ticker = f"{ticker}.NS"
                elif exchange in ['BSE', 'BOM']:
                    ticker = f"{ticker}.BO"
                
                print(f"[TRADINGVIEW] '{company_name}' -> {ticker} ({description})")
                return ticker
            
            return None
            
        except Exception as e:
            print(f"[TRADINGVIEW ERROR] {str(e)}")
            return None
    
    def _search_yahoo(self, company_name: str) -> Optional[str]:
        """
        Search using Yahoo Finance API (Fallback)
        """
        try:
            url = "https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                'q': company_name,
                'quotesCount': 5,
                'newsCount': 0,
                'enableFuzzyQuery': False,
                'quotesQueryId': 'tss_match_phrase_query'
            }
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            data = response.json()
            
            quotes = data.get('quotes', [])
            if quotes:
                best_match = quotes[0]
                symbol = best_match.get('symbol')
                print(f"[YAHOO FALLBACK] '{company_name}' -> {symbol} ({best_match.get('shortname', 'N/A')})")
                return symbol
            
            return None
            
        except Exception as e:
            print(f"[YAHOO ERROR] {str(e)}")
            return None

    def search_ticker_symbol(self, company_name: str) -> Optional[str]:
        """
        Search for ticker symbol using TradingView (primary) + Yahoo Finance (fallback)
        Returns the best matching ticker symbol or None
        """
        if company_name in self.ticker_cache:
            print(f"[CACHE HIT] '{company_name}' -> {self.ticker_cache[company_name]}")
            return self.ticker_cache[company_name]
        
        symbol = self._search_tradingview(company_name)
        
        if not symbol:
            print(f"[FALLBACK] Trying Yahoo Finance for '{company_name}'...")
            symbol = self._search_yahoo(company_name)
        
        if symbol:
            self.ticker_cache[company_name] = symbol
        else:
            print(f"[SEARCH FAILED] No ticker found for '{company_name}'")
        
        return symbol

    def _build_graph(self)->StateGraph:
        """Build with Langgraph Workflow"""

        #nodes
        workflow=StateGraph(AgentState)
        workflow.add_node("parse_query",self.parse_query_node)
        workflow.add_node("plan_analysis",self.plan_analysis_node)
        workflow.add_node("execute_tools",self.execute_tools_node)
        workflow.add_node("synthesize",self.synthesize_node)
        workflow.add_node("format_response",self.format_response_node)

        #edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query","plan_analysis")
        workflow.add_edge("plan_analysis","execute_tools")
        workflow.add_edge("execute_tools","synthesize")
        workflow.add_edge("synthesize","format_response")
        workflow.add_edge("format_response",END)
        
        return workflow.compile()

    def parse_query_node(self,state:AgentState)->AgentState:
        """
        Node 1: Parse user query
        Extract symbols,determine intent using LLM + TradingView (primary) + Yahoo Finance (fallback)
        """   
        query=state["user_query"]
        prompt=f"""You are a financial assistant. Analyze this query and extract:
1. Company/stock names mentioned
2. Query type (price, risk, sentiment, investment_decision, news)
3. User intent

Query: "{query}"

Return ONLY valid JSON (no markdown):
{{"company_names":["Bank of Maharashtra"],"query_type":"price","intent":"monthly performance analysis"}}

If ticker symbols are already provided (e.g., AAPL, SBIN.NS), put them in company_names as-is.
"""
        response=self.llm.invoke(prompt)
        import re
        import json
        try:
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(item.get('text', str(item)))
                    else:
                        text_parts.append(str(item))            
            content = ' '.join(text_parts)
            
            json_match = re.search(r'\{[^\{\}]*"company_names"[^\{\}]*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                company_names = parsed.get("company_names", [])
                state["query_type"] = parsed.get("query_type", "investment_decision")
                state["intent"] = parsed.get("intent", query)
                
                resolved_symbols = []
                for name in company_names:
                    # Check abbreviation map first
                    name_upper = name.upper()
                    if name_upper in self.abbreviation_map:
                        resolved_symbols.append(self.abbreviation_map[name_upper])
                        print(f"[ABBREVIATION] '{name}' -> {self.abbreviation_map[name_upper]}")
                    elif re.match(r'^[A-Z]{1,5}(?:\.[A-Z]{1,3})?$', name_upper):
                        resolved_symbols.append(name_upper)
                        print(f"[TICKER] Using provided symbol: {name_upper}")
                    else:
                        symbol = self.search_ticker_symbol(name)
                        if symbol:
                            resolved_symbols.append(symbol)
                        else:
                            print(f"[WARNING] Could not resolve '{name}', skipping")
                
                state["symbols"] = resolved_symbols if resolved_symbols else ["AAPL"]
                
            else:
                symbol = self.search_ticker_symbol(query)
                if symbol:
                    state["symbols"] = [symbol]
                else:
                    symbols = re.findall(r'\b[A-Z]{2,5}(?:\.(?:NS|BO))?\b', query.upper())
                    state["symbols"] = symbols if symbols else ["AAPL"]
                
                state["query_type"] = "investment_decision"
                state["intent"] = query
                
                state["query_type"] = "sentiment" if "sentiment" in query.lower() else "investment_decision"
                state["intent"] = query
        except Exception as e:
            state["symbols"] = ["AAPL"]
            state["query_type"] = "investment_decision"
            state["intent"] = query
        
        state["step_count"]=1 
        return state

    def plan_analysis_node(self,state:AgentState)->AgentState:
        """
        Node 2: Plan which tools to use
        Based on query type, decide tool sequence
        """
        query_type=state["query_type"]
        symbols = state["symbols"]
        if len(symbols) > 1 and query_type == "investment_decision":
            state["tools_to_use"]=["compare_stocks"]
        elif query_type=="price":
            state["tools_to_use"]=["get_stock_price", "get_stock_news"]
        elif query_type=="risk":
            state["tools_to_use"]=["get_analyze_risk", "get_stock_info", "get_stock_news", "analyze_news_sentiment"]
        elif query_type=="sentiment":
            state["tools_to_use"]=["get_stock_price", "get_stock_news", "analyze_news_sentiment"]
        elif query_type=="news":
            state["tools_to_use"]=["get_stock_news", "analyze_news_sentiment"]
        elif query_type == "investment_decision":
            state["tools_to_use"]=[
                "get_stock_price",
                "get_analyze_risk",
                "predict_price",
                "get_stock_news",
                "analyze_news_sentiment",
                "get_financial_metrics"
            ]
        else:
            state["tools_to_use"]=["get_stock_price", "get_stock_news"]

        state["step_count"]+=1
        return state
    
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """Node 3: Execute tools"""
        tools_to_use = state["tools_to_use"]
        symbols = state["symbols"] if state["symbols"] else ["AAPL"]
        results = {}
        if "compare_stocks" in tools_to_use:
            try:
                tool = next((t for t in ALL_TOOLS if t.name == "compare_stocks"), None)
                if tool:
                    result = await tool.ainvoke({"symbols": symbols})
                    results["compare_stocks"] = result
            except Exception as e:
                results["compare_stocks"] = {"error": str(e)}
        else:
            symbol = symbols[0]
            company_name = symbol.replace('.NS', '').replace('.BO', '')
            symbol_to_name = {
                "SBIN": "State Bank of India",
                "RELIANCE": "Reliance Industries",
                "TCS": "Tata Consultancy Services",
                "INFY": "Infosys",
                "HDFCBANK": "HDFC Bank",
                "ICICIBANK": "ICICI Bank",
                "AAPL": "Apple Inc"
            }
            full_company_name = symbol_to_name.get(company_name, company_name)
            
            for tool_name in tools_to_use:
                try:
                    tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)
                    if tool:
                        if tool_name in ["get_stock_news", "analyze_news_sentiment"]:
                            result = await tool.ainvoke({
                                "symbol": symbol,
                                "company_name": full_company_name
                            })
                        else:
                            result = await tool.ainvoke({"symbol": symbol})
                        
                        results[tool_name] = result
                except Exception as e:
                    results[tool_name] = {"error": str(e)}
        
        state["tool_results"] = results
        state["step_count"] += 1
        return state
    
    def synthesize_node(self, state: AgentState) -> AgentState:
        """Node 4: Synthesize results"""
        import json
        tool_results = state["tool_results"]
        symbol = state["symbols"][0] if state["symbols"] else "Unknown"
        query_type = state["query_type"]
        parsed_results = {}
        for tool_name, result in tool_results.items():
            if isinstance(result, str):
                try:
                    parsed_results[tool_name] = json.loads(result)
                except:
                    parsed_results[tool_name] = {"raw": result}
            else:
                parsed_results[tool_name] = result
        news_data = parsed_results.get("get_stock_news", {})
        sentiment_data = parsed_results.get("analyze_news_sentiment", {})
        
        news_articles = []
        if isinstance(news_data, list):
            for article in news_data[:5]:
                if isinstance(article, dict):
                    news_articles.append({
                        'title': article.get('title', 'N/A'),
                        'url': article.get('url', ''),
                        'source': article.get('source', 'Unknown'),
                        'sentiment': article.get('sentiment', 'neutral')
                    })
        
        prompt = f"""You are a financial analyst. Analyze the following data for {symbol}:

Query Type: {query_type}

Financial Data:
{json.dumps(parsed_results, indent=2)}

News Articles (WITH LINKS):
{json.dumps(news_articles, indent=2)}

CRITICAL REQUIREMENTS - YOU MUST INCLUDE ALL OF THESE:

1. **Key Insights** (3-5 bullet points with specific numbers from the data)

2. **Recommendation**: BUY, HOLD, or SELL with confidence level (high/medium/low)

3. **Detailed Reasoning** (2-3 sentences):
   - Reference SPECIFIC financial metrics (price, volatility, P/E ratio, etc.)
   - Mention sentiment analysis results
   - Cite at least 2 news events

4. **Supporting News References** (MANDATORY - MUST INCLUDE 3-5):
   Format each as:
   • [Headline] - Source
     Link: [URL]
     Relevance: [1 sentence explaining how this supports your recommendation]

Example format:
📰 **Supporting News & Reports:**
• ICICI Bank reports strong Q3 earnings - Economic Times
  Link: https://example.com/article1
  Relevance: Strong earnings support bullish outlook

• RBI maintains interest rates - Reuters
  Link: https://example.com/article2
  Relevance: Stable rates benefit banking sector profitability

IMPORTANT: You MUST include actual URLs from the news_articles data above. Do not skip the news references section."""
        
        response=self.llm.invoke(prompt)
        content=response.content
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        if "buy" in content.lower() and "don't buy" not in content.lower() and "not buy" not in content.lower():
            state["recommendation"] = "BUY"
        elif "sell" in content.lower() and "don't sell" not in content.lower():
            state["recommendation"] = "SELL"
        else:
            state["recommendation"] = "HOLD"
        
        if "high confidence" in content.lower() or "strong" in content.lower():
            state["confidence"] = "high"
        elif "low confidence" in content.lower() or "uncertain" in content.lower():
            state["confidence"] = "low"
        else:
            state["confidence"] = "medium"
        insights = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0:2].replace('.','').isdigit()):
                insights.append(line.lstrip('-•*0123456789. '))
        
        state["insights"] = insights if insights else [content[:200] + "..."]
        state["full_analysis"] = content
        headlines = []
        if isinstance(news_data, list):
            headlines = news_data[:5]
        sentiment = "neutral"
        sentiment_score = 0.0
        if isinstance(sentiment_data, dict):
            sentiment = sentiment_data.get("overall_sentiment", sentiment_data.get("sentiment", "neutral"))
            sentiment_score = sentiment_data.get("score", sentiment_data.get("sentiment_score", 0.0))
        
        state["news_references"] = {
            "headlines": headlines[:3] if isinstance(headlines, list) else [], 
            "sentiment": sentiment,
            "sentiment_score": sentiment_score
        }
        
        state["step_count"] += 1
        return state
    
    def format_response_node(self,state:AgentState)->AgentState:
        """
        Node 5: Format final response
        Create user-friendly output
        """
        symbol=state["symbols"][0] if state["symbols"] else "Unknown"
        recommendation=state["recommendation"]  
        confidence=state["confidence"]
        full_analysis = state.get("full_analysis", "")
        news_refs = state.get("news_references", {})        
        news_section = ""
        if news_refs:
            headlines = news_refs.get("headlines", [])
            sentiment = news_refs.get("sentiment", "neutral")
            score = news_refs.get("sentiment_score", 0)
            
            if headlines and len(headlines) > 0:
                news_section = f"\n\n### 📰 News References Supporting This Decision\n\n"
                news_section += f"**Market Sentiment:** {sentiment.upper()} (Score: {score:.2f})\n\n"
                news_section += "**Top 3 News Articles:**\n"
                for i, headline in enumerate(headlines[:3], 1):  # Show exactly 3 references
                    if isinstance(headline, dict):
                        title = headline.get('title', headline.get('headline', headline.get('description', 'N/A')))
                        source = headline.get('source', headline.get('publisher', headline.get('author', 'Unknown')))
                        url = headline.get('url', headline.get('link', '#'))
                        published = headline.get('publishedAt', headline.get('date', ''))
                        
                        news_section += f"\n{i}. **{title}**\n"
                        news_section += f"   - Source: *{source}*\n"
                        if published:
                            news_section += f"   - Date: {published[:10]}\n"
                        if url and url != '#':
                            news_section += f"   - Link: {url}\n"
                    else:
                        news_section += f"{i}. {headline}\n"
                        
                news_section += "\n*These articles from News API were used to inform the recommendation.*"
            else:
                news_section = "\n\n### 📰 News References\n\n*No recent news articles found. Recommendation based on technical analysis only.*"
        
        response=f"""📊 **Analysis for {symbol}**

**Recommendation:** {recommendation} (Confidence: {confidence.upper()})

{full_analysis}{news_section}

---
*Note: This analysis is based on current market data and news sentiment. This should not be considered financial advice. Always do your own research before investing.*
        """ 
        state["messages"].append({
            "role":"assistant",
            "content":response
        })
        state["should_continue"]=False
        state["step_count"]+=1
        return state
    
    async def run(self, query: str) -> Dict:
        """
        Run the agent on a query
        Args:
            query: User query string
        Returns:
            Final state dictionary
        """ 
        initial_state={
            "messages":[{"role": "user", "content": query}],
            "user_query":query,
            "symbols":[],
            "query_type":"",
            "intent":"",
            "tools_to_use":[],
            "tool_results":{},
            "market_data":{},
            "risk_analysis":{},
            "news_sentiment":{},
            "predictions":{},
            "insights":[],
            "recommendation":None,
            "confidence":"",
            "full_analysis":"",
            "news_references":{},
            "should_continue":True,
            "error":None,
            "step_count":0            
        }
        final_state=await self.graph.ainvoke(initial_state)
        return final_state



