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
        self.abbreviation_map = {
            "SBI": "SBIN.NS",
            "HDFC": "HDFCBANK.NS",
            "ICICI": "ICICIBANK.NS",
            "BOB": "BANKBARODA.NS",
            "PNB": "PNB.NS",
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
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
        Search using Yahoo Finance API with smart search variations
        """
        try:
            search_query = company_name.strip()
            if not any(ex in search_query.upper() for ex in ['.NS', '.BO', 'NASDAQ', 'NYSE']):
                search_variations = [
                    search_query,
                    f"{search_query} NSE",
                    f"{search_query} India",
                ]
            else:
                search_variations = [search_query]
            
            for query in search_variations:
                url = "https://query2.finance.yahoo.com/v1/finance/search"
                params = {
                    'q': query,
                    'quotesCount': 5,
                    'newsCount': 0,
                    'enableFuzzyQuery': True, 
                    'quotesQueryId': 'tss_match_phrase_query'
                }
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                response = requests.get(url, params=params, headers=headers, timeout=5)
                data = response.json()
                
                quotes = data.get('quotes', [])
                if quotes:
                    indian_quotes = [q for q in quotes if '.NS' in q.get('symbol', '') or '.BO' in q.get('symbol', '')]
                    best_match = indian_quotes[0] if indian_quotes else quotes[0]
                    
                    symbol = best_match.get('symbol')
                    name = best_match.get('shortname', best_match.get('longname', ''))
                    print(f"[YAHOO] '{company_name}' -> {symbol} ({name})")
                    return symbol
            
            return None
            
        except Exception as e:
            print(f"[YAHOO ERROR] {str(e)}")
            return None

    def search_ticker_symbol(self, company_name: str) -> Optional[str]:
        """
        Search for ticker symbol using Yahoo (primary) + TradingView (fallback)
        Returns the best matching ticker symbol or None
        """
        if company_name in self.ticker_cache:
            print(f"[CACHE HIT] '{company_name}' -> {self.ticker_cache[company_name]}")
            return self.ticker_cache[company_name]
        symbol = self._search_yahoo(company_name)
        if not symbol:
            print(f"[FALLBACK] Trying TradingView for '{company_name}'...")
            symbol = self._search_tradingview(company_name)
        
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
        Extract company names and use AI-powered search to find correct ticker symbols
        """   
        query=state["user_query"]
        prompt=f"""You are a financial assistant. Analyze this query and extract:
1. Company/stock names mentioned (use BRAND NAMES, not legal names)
2. Query type (price, risk, sentiment, investment_decision, news)
3. User intent

Query: "{query}"

Return ONLY valid JSON (no markdown):
{{"company_names":["Bank of Maharashtra"],"query_type":"price","intent":"monthly performance analysis"}}

IMPORTANT RULES:
- Use BRAND NAMES: "Nykaa" not "FSN E-Commerce Ventures Ltd."
- Use BRAND NAMES: "Bank of Baroda" not "Bank of Baroda Ltd."
- Use BRAND NAMES: "Paytm" not "One 97 Communications"
- For typos (e.g., "nykka"), correct to proper brand name (e.g., "Nykaa")
- Do NOT add .NS or .BO suffix - just the brand name
- If user provides ticker with suffix (e.g., SBIN.NS), keep it as-is
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
                    name_upper = name.upper()
                    is_ticker_with_exchange = '.' in name_upper  
                    is_short_us_ticker = re.match(r'^[A-Z]{1,4}$', name_upper) and not name_upper in ['NYKAA', 'PAYTM', 'ZOMATO']  # Short US tickers
                    is_known_ticker = name_upper in self.abbreviation_map
                    
                    if is_ticker_with_exchange or is_known_ticker:
                        if is_known_ticker:
                            resolved_symbols.append(self.abbreviation_map[name_upper])
                            print(f"[ABBREVIATION] '{name}' -> {self.abbreviation_map[name_upper]}")
                        else:
                            resolved_symbols.append(name_upper)
                            print(f"[TICKER] Using provided symbol: {name_upper}")
                    else:
                        print(f"[SEARCH] Looking up ticker for: {name}")
                        symbol = self.search_ticker_symbol(name)
                        if symbol:
                            resolved_symbols.append(symbol)
                            print(f"[RESOLVED] '{name}' -> {symbol}")
                        else:
                            print(f"[WARNING] Could not resolve '{name}', trying fallback...")
                            fallback_symbol = self.search_ticker_symbol(f"{name} India")
                            if fallback_symbol:
                                resolved_symbols.append(fallback_symbol)
                                print(f"[FALLBACK] '{name}' -> {fallback_symbol}")
                            else:
                                print(f"[ERROR] Failed to find ticker for '{name}'")
                if not resolved_symbols:
                    print(f"[FALLBACK] No symbols found, trying to search entire query...")
                    symbol = self.search_ticker_symbol(query)
                    state["symbols"] = [symbol] if symbol else ["AAPL"]
                else:
                    state["symbols"] = resolved_symbols
                
            else:
                print(f"[FALLBACK] Could not parse JSON, searching query directly...")
                symbol = self.search_ticker_symbol(query)
                if symbol:
                    state["symbols"] = [symbol]
                else:
                    symbols = re.findall(r'\b[A-Z]{2,5}(?:\.(?:NS|BO))?\b', query.upper())
                    state["symbols"] = symbols if symbols else ["AAPL"]
                
                state["query_type"] = "sentiment" if "sentiment" in query.lower() else "investment_decision"
                state["intent"] = query
                
        except Exception as e:
            print(f"[ERROR] Parse error: {e}")
            symbol = self.search_ticker_symbol(query)
            state["symbols"] = [symbol] if symbol else ["AAPL"]
            state["query_type"] = "investment_decision"
            state["intent"] = query
        
        print(f"\n[FINAL] Resolved symbols: {state['symbols']}")
        print(f"[FINAL] Query type: {state['query_type']}\n")
        
        state["step_count"]=1 
        return state

    def plan_analysis_node(self,state:AgentState)->AgentState:
        """
        Node 2: LLM-powered analysis planning
        Let AI decide which tools to use based on the query
        """
        query = state["user_query"]
        intent = state.get("intent", query)
        symbols = state["symbols"]
        if len(symbols) > 1:
            state["tools_to_use"] = ["compare_stocks"]
            state["step_count"] += 1
            return state
        prompt = f"""You are a financial analysis planner. Based on the user's query, select the most relevant tools to gather information.

User Query: "{query}"
Intent: "{intent}"
Stock Symbol: {symbols[0] if symbols else "Unknown"}

Available Tools:
1. get_stock_price - Get current price, market cap, P/E ratio
2. get_financial_metrics - Get profitability, valuation ratios, growth metrics
3. get_analyze_risk - Calculate volatility, VaR, risk scoring
4. predict_price - Predict future price using EMA/Linear regression
5. analyze_chart - AI analysis of chart patterns and technical signals
6. get_stock_news - Fetch recent news articles
7. analyze_news_sentiment - Analyze sentiment of news (positive/negative/neutral)
8. summarize_news_articles - AI-generated summary of all news

Select 3-8 tools that would be most useful for this query.

Return ONLY a JSON array of tool names (no markdown, no explanation):
["get_stock_price", "get_financial_metrics", "get_stock_news"]

Guidelines:
- Always include get_stock_price for basic info
- For investment decisions: include financial_metrics, risk, prediction, chart, news, sentiment
- For price queries: include price, chart, news
- For risk queries: include risk, financial_metrics, chart
- For sentiment queries: include news, sentiment, summarize_news_articles
- For news queries: include news, sentiment, summarize_news_articles
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            import re
            json_match = re.search(r'\[([^\[\]]*)\]', content)
            if json_match:
                json_str = f"[{json_match.group(1)}]"
                tools = json.loads(json_str)
                valid_tools = [
                    "get_stock_price", "get_financial_metrics", "get_analyze_risk",
                    "predict_price", "analyze_chart", "get_stock_news",
                    "analyze_news_sentiment", "summarize_news_articles"
                ]
                
                selected_tools = [t for t in tools if t in valid_tools]
                
                if selected_tools:
                    state["tools_to_use"] = selected_tools
                    print(f"\n[AI PLANNER] Selected {len(selected_tools)} tools: {', '.join(selected_tools)}\n")
                else:
                    state["tools_to_use"] = ["get_stock_price", "get_stock_news"]
                    print("[AI PLANNER] Using fallback tools")
            else:
                state["tools_to_use"] = ["get_stock_price", "get_stock_news"]
                print("[AI PLANNER] JSON parse failed, using fallback tools")
                
        except Exception as e:
            print(f"[AI PLANNER ERROR] {str(e)}")
            state["tools_to_use"] = [
                "get_stock_price",
                "get_financial_metrics",
                "get_analyze_risk",
                "predict_price",
                "analyze_chart",
                "get_stock_news",
                "analyze_news_sentiment",
                "summarize_news_articles"
            ]
        
        state["step_count"] += 1
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
                        if tool_name in ["get_stock_news", "analyze_news_sentiment", "summarize_news_articles"]:
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
   - USE CORRECT CURRENCY SYMBOL based on stock exchange:
     * .NS or .BO (India) → ₹
     * .L (London) → £
     * .PA, .DE, .AS, .MI, .MC (Europe) → €
     * .T (Tokyo) → ¥
     * .HK (Hong Kong) → HK$
     * No suffix or US exchanges → $

2. **Recommendation**: BUY, HOLD, or SELL with confidence level (high/medium/low)

3. **Detailed Reasoning** (2-3 sentences):
   - Reference SPECIFIC financial metrics (price, volatility, P/E ratio, etc.) with CORRECT CURRENCY
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

IMPORTANT: 
- Always use the CORRECT currency symbol in all monetary values
- You MUST include actual URLs from the news_articles data above
- Do not skip the news references section"""
        
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
        Node 5: Format final response with comprehensive data display
        """
        import json
        symbol=state["symbols"][0] if state["symbols"] else "Unknown"
        recommendation=state["recommendation"]  
        confidence=state["confidence"]
        full_analysis = state.get("full_analysis", "")
        tool_results = state.get("tool_results", {})
        news_refs = state.get("news_references", {})
        detailed_sections = []
        price_data = tool_results.get("get_stock_price", {})
        if isinstance(price_data, str):
            try:
                price_data = json.loads(price_data)
            except:
                price_data = {}
        
        if price_data and not price_data.get("error"):
            detailed_sections.append(f"""
### 📈 Current Market Data

- **Current Price:** {price_data.get('current_price', 'N/A')}
- **Market Cap:** {price_data.get('market_cap', 0):,.0f}
- **P/E Ratio:** {price_data.get('pe_ratio', 'N/A')}
- **Day Change:** {price_data.get('change', 0):.2f} ({price_data.get('change_percent', 0):.2f}%)
- **Day Range:** {price_data.get('low', 'N/A')} - {price_data.get('high', 'N/A')}
- **Volume:** {price_data.get('volume', 0):,.0f}
- **Sector:** {price_data.get('sector', 'N/A')}
""")
        metrics_data = tool_results.get("get_financial_metrics", {})
        if isinstance(metrics_data, str):
            try:
                metrics_data = json.loads(metrics_data)
            except:
                metrics_data = {}
        
        if metrics_data and not metrics_data.get("error"):
            detailed_sections.append(f"""
### 💰 Financial Metrics

**Profitability:**
- Profit Margin: {metrics_data.get('profit_margin', 0):.2f}%
- Operating Margin: {metrics_data.get('operating_margin', 0):.2f}%
- ROE (Return on Equity): {metrics_data.get('roe', 0):.2f}%
- ROA (Return on Assets): {metrics_data.get('roa', 0):.2f}%

**Valuation Ratios:**
- P/E Ratio: {metrics_data.get('pe_ratio', 'N/A')}
- Forward P/E: {metrics_data.get('forward_pe', 'N/A')}
- Price to Book: {metrics_data.get('price_to_book', 'N/A')}
- PEG Ratio: {metrics_data.get('peg_ratio', 'N/A')}

**Growth:**
- Revenue Growth: {metrics_data.get('revenue_growth', 0):.2f}%
- Earnings Growth: {metrics_data.get('earnings_growth', 0):.2f}%

**Financial Health:**
- Debt to Equity: {metrics_data.get('debt_to_equity', 'N/A')}
- Current Ratio: {metrics_data.get('current_ratio', 'N/A')}
""")
        risk_data = tool_results.get("get_analyze_risk", {})
        if isinstance(risk_data, str):
            try:
                risk_data = json.loads(risk_data)
            except:
                risk_data = {}
        
        if risk_data and not risk_data.get("error"):
            detailed_sections.append(f"""
### ⚠️ Risk Analysis

- **Volatility (Annualized):** {risk_data.get('volatility', 0):.2%}
- **Max Drawdown:** {risk_data.get('max_drawdown', 0):.2%}
- **Sharpe Ratio:** {risk_data.get('sharpe_ratio', 'N/A')}
- **VaR (95%):** {risk_data.get('var_95', 0):.2f}
- **Beta:** {risk_data.get('beta', 'N/A')}
- **Risk Level:** {risk_data.get('risk_level', 'N/A').upper()}
""")
        prediction_data = tool_results.get("predict_price", {})
        if isinstance(prediction_data, str):
            try:
                prediction_data = json.loads(prediction_data)
            except:
                prediction_data = {}
        
        if prediction_data and not prediction_data.get("error"):
            detailed_sections.append(f"""
### 🔮 Price Prediction

- **Predicted Price (Next Period):** {prediction_data.get('predicted_price', 'N/A')}
- **Current Price:** {prediction_data.get('current_price', 'N/A')}
- **Expected Change:** {prediction_data.get('predicted_change_percent', 0):.2f}%
- **Method Used:** {prediction_data.get('method', 'N/A').upper()}
- **Confidence:** {prediction_data.get('confidence', 'N/A')}
""")
        sentiment_data = tool_results.get("analyze_news_sentiment", {})
        if isinstance(sentiment_data, str):
            try:
                sentiment_data = json.loads(sentiment_data)
            except:
                sentiment_data = {}
        
        if sentiment_data and not sentiment_data.get("error"):
            detailed_sections.append(f"""
### 📊 News Sentiment Analysis

- **Overall Sentiment:** {sentiment_data.get('overall_sentiment', 'N/A').upper()}
- **Sentiment Score:** {sentiment_data.get('sentiment_score', 0):.2f}
- **Confidence:** {sentiment_data.get('confidence', 'N/A').upper()}
- **Articles Analyzed:** {sentiment_data.get('total_articles', 0)}
  - Positive: {sentiment_data.get('positive_count', 0)}
  - Negative: {sentiment_data.get('negative_count', 0)}
  - Neutral: {sentiment_data.get('neutral_count', 0)}
- **Summary:** {sentiment_data.get('summary', 'N/A')}
""")
        chart_data = tool_results.get("analyze_chart", {})
        if isinstance(chart_data, str):
            try:
                chart_data = json.loads(chart_data)
            except:
                chart_data = {}
        
        if chart_data and not chart_data.get("error"):
            detailed_sections.append(f"""
### 📈 Technical Chart Analysis

{chart_data.get('analysis', 'N/A')}
""")
        news_summary_data = tool_results.get("summarize_news_articles", {})
        if isinstance(news_summary_data, str):
            try:
                news_summary_data = json.loads(news_summary_data)
            except:
                news_summary_data = {}
        
        if news_summary_data and not news_summary_data.get("error"):
            detailed_sections.append(f"""
### 📰 AI-Generated News Summary

        {news_summary_data.get('summary', 'N/A')}
""")
        news_section = ""  
        if news_refs:
            headlines = news_refs.get("headlines", [])
            sentiment = news_refs.get("sentiment", "neutral")
            score = news_refs.get("sentiment_score", 0)
            
            if headlines and len(headlines) > 0:
                news_section = f"\n\n### 📰 Recent News Articles\n\n"
                news_section += f"**Market Sentiment:** {sentiment.upper()} (Score: {score:.2f})\n\n"
                for i, headline in enumerate(headlines[:5], 1):
                    if isinstance(headline, dict):
                        title = headline.get('title', headline.get('headline', headline.get('description', 'N/A')))
                        source = headline.get('source', headline.get('publisher', headline.get('author', 'Unknown')))
                        url = headline.get('url', headline.get('link', '#'))
                        published = headline.get('publishedAt', headline.get('date', ''))
                        
                        news_section += f"\n**{i}. {title}**\n"
                        news_section += f"   - Source: {source}"
                        if published:
                            news_section += f" | Date: {published[:10]}"
                        news_section += "\n"
                        if url and url != '#':
                            news_section += f"   - Link: {url}\n"
                    else:
                        news_section += f"{i}. {headline}\n"
        detailed_data = "\n".join(detailed_sections) if detailed_sections else "\n*Detailed financial data unavailable.*\n"
        
        response=f"""📊 **Comprehensive Analysis for {symbol}**

**Recommendation:** {recommendation} (Confidence: {confidence.upper()})

---

## 🎯 Executive Summary

{full_analysis}

---

## 📋 Detailed Financial Data
{detailed_data}
{news_section}

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



