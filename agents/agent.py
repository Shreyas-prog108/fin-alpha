from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Tuple, Optional
import json

from .state import AgentState
from .config import config
from .tools import ALL_TOOLS
from .clients import get_tradingview_client, get_backend_client
from .prompts import (
    MAIN_AGENT_SYSTEM_PROMPT,
    TOOL_SELECTION_GUIDE,
    SYNTHESIS_PROMPT,
)
from .prompts.subagent_prompts import (
    MARKET_DATA_AGENT_PROMPT,
    RISK_AGENT_PROMPT,
    SENTIMENT_AGENT_PROMPT,
    PREDICTION_AGENT_PROMPT
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
        self.symbol_metadata_cache = {}
        self.tradingview = get_tradingview_client()
        self.backend = get_backend_client()

    def _normalize_timeframe(self, timeframe: Optional[str]) -> str:
        allowed = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
        if not timeframe:
            return "1mo"
        tf = timeframe.strip().lower()
        if tf in allowed:
            return tf
        mapping = {
            "today": "1d",
            "day": "1d",
            "daily": "1d",
            "week": "5d",
            "weekly": "5d",
            "month": "1mo",
            "monthly": "1mo",
            "quarter": "3mo",
            "3 months": "3mo",
            "six months": "6mo",
            "6 months": "6mo",
            "half year": "6mo",
            "year": "1y",
            "yearly": "1y",
            "2 years": "2y",
            "5 years": "5y",
            "long term": "5y",
            "all": "max",
            "all time": "max"
        }
        return mapping.get(tf, "1mo")

    def _normalize_sentiment_focus(self, sentiment: Optional[str]) -> str:
        if not sentiment:
            return "unknown"
        s = sentiment.strip().lower()
        if s in ["bullish", "positive"]:
            return "positive"
        if s in ["bearish", "negative"]:
            return "negative"
        if s in ["neutral", "mixed"]:
            return s
        return "unknown"

    def _normalize_news_category(self, category: Optional[str]) -> str:
        if not category:
            return "general"
        c = category.strip().lower()
        if c in ["general", "all", "any"]:
            return "general"
        return c

    def _infer_timeframe_from_query(
        self,
        query: str,
        intent: str,
        sentiment_focus: str
    ) -> str:
        prompt = f"""You are a financial assistant. Infer the most appropriate Yahoo period
for historical analysis based on the user's sentiment and intent.

User Query: "{query}"
Intent: "{intent}"
Sentiment Focus: "{sentiment_focus}"

Return ONLY one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max

Guidance:
- Urgent/short-term concern or high volatility focus -> 1d or 5d
- Mixed/neutral sentiment with general analysis -> 1mo
- Long-term investment/growth focus -> 6mo to 1y
- Very long-term/retirement -> 2y or 5y
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            if not isinstance(content, str):
                content = str(content)
            candidate = content.strip().lower()
            allowed = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
            if candidate in allowed:
                return candidate
        except Exception as e:
            print(f"[TIMEFRAME INFER ERROR] {str(e)}")

        return "1mo"

    def _timeframe_to_days(self, timeframe: str) -> int:
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "max": 1825
        }
        return mapping.get(timeframe, 30)

    def _get_company_name(
        self,
        symbol: str,
        symbol_metadata: Optional[Dict[str, Dict]] = None
    ) -> str:
        if symbol_metadata and symbol in symbol_metadata:
            name = symbol_metadata[symbol].get("name")
            if name:
                return name
        cached = self.symbol_metadata_cache.get(symbol, {})
        if cached.get("name"):
            return cached["name"]
        return symbol.replace(".NSE", "").replace(".BO", "")

    def _apply_exchange_suffix(self, ticker: str, exchange: str) -> str:
        if not ticker:
            return ""
        if "." in ticker:
            return ticker.upper()
        exchange_upper = (exchange or "").upper()
        if exchange_upper in ["NSE", "NSI"]:
            return f"{ticker.upper()}.NSE"
        if exchange_upper in ["BSE", "BOM"]:
            return f"{ticker.upper()}.BO"
        return ticker.upper()

    def _looks_like_ticker(self, value: str) -> bool:
        import re
        if not value:
            return False
        return bool(re.match(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,3})?$", value.upper()))

    def _parse_tool_result(self, result: object) -> Dict:
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return {"raw": result}
        if isinstance(result, dict):
            return result
        return {"raw": result}

    def _resolve_symbol_from_tradingview(self, query: str) -> Tuple[Optional[str], Dict]:
        """TradingView symbol resolution (often blocked by rate limits)"""
        try:
            result = self.tradingview.search_symbol(query)
            if not result:
                return None, {}
            ticker = result.get("ticker", "")
            exchange = result.get("exchange", "")
            description = result.get("description", "") or query
            symbol = self._apply_exchange_suffix(ticker, exchange)
            metadata = {
                "name": description,
                "exchange": exchange,
                "source": "tradingview"
            }
            return symbol, metadata
        except Exception as e:
            print(f"[TRADINGVIEW ERROR] {str(e)}")
            return None, {}

    def _resolve_symbol_fallback(self, query: str) -> Tuple[Optional[str], Dict]:
        """
        Minimal fallback for symbol resolution.
        Only used when Gemini doesn't provide yahoo_symbols.
        Just formats the input - no hardcoded stock lists.
        """
        query_clean = query.strip()
        if not query_clean:
            return None, {}
        
        query_upper = query_clean.upper()
        

        if any(suffix in query_upper for suffix in ['.NSE','.BSE', '.BO', '.L', '.HK', '.T', '.AX', '.TO']):
            exchange = "NSE" if ".NSE" in query_upper else "Other"
            return query_upper, {"name": query_clean, "exchange": exchange, "source": "direct"}
        

        if len(query_upper) <= 15 and query_upper.replace("&", "").replace("-", "").replace(".", "").isalnum():
            return query_upper, {"name": query_clean, "exchange": "Unknown", "source": "fallback"}
        
        return None, {}

    def resolve_symbol(self, query: str) -> Tuple[Optional[str], Dict]:
        """
        Minimal symbol resolution fallback.
        Gemini should handle all symbol resolution in parse_query_node.
        This is only used as a last resort.
        """
        query_clean = query.strip()
        if not query_clean:
            return None, {}
        

        cache_key = query_clean.lower()
        if cache_key in self.ticker_cache:
            symbol = self.ticker_cache[cache_key]
            print(f"[CACHE HIT] '{query_clean}' -> {symbol}")
            return symbol, self.symbol_metadata_cache.get(symbol, {})


        symbol, metadata = self._resolve_symbol_fallback(query_clean)
        if symbol:
            self.ticker_cache[cache_key] = symbol
            if metadata:
                self.symbol_metadata_cache[symbol] = metadata
            print(f"[FALLBACK] '{query_clean}' -> {symbol}")
            return symbol, metadata

        print(f"[SEARCH FAILED] No symbol found for '{query_clean}'")
        return None, {}

    def _build_graph(self)->StateGraph:
        """Build with Langgraph Workflow"""

        #nodes
        workflow=StateGraph(AgentState)
        workflow.add_node("parse_query", self.parse_query_node)
        workflow.add_node("prefetch_context", self.prefetch_context_node)
        workflow.add_node("plan_analysis", self.plan_analysis_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("run_subagents", self.run_subagents_node)
        workflow.add_node("synthesize", self.synthesize_node)
        workflow.add_node("format_response", self.format_response_node)

        #edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "prefetch_context")
        workflow.add_edge("prefetch_context", "plan_analysis")
        workflow.add_edge("plan_analysis", "execute_tools")
        workflow.add_edge("execute_tools", "run_subagents")
        workflow.add_edge("run_subagents", "synthesize")
        workflow.add_edge("synthesize", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()

    async def parse_query_node(self,state:AgentState)->AgentState:
        """
        Node 1: Parse user query
        Use Gemini to extract stock symbols directly in Yahoo Finance format
        """   
        query=state["user_query"]
        prompt=f"""You are a financial assistant expert in stock markets. Analyze this query and extract information.

USER QUERY: "{query}"

Extract the following and return ONLY a valid JSON object:

1. **yahoo_symbols**: The EXACT Yahoo Finance ticker symbols for the stocks mentioned.
   - For Indian stocks on NSE: Add ".NSE" suffix (e.g., "SBIN.NSE" for SBI, "RELIANCE.NSE" for Reliance, "TCS.NSE" for TCS)
   - For Indian stocks on BSE: Add ".BO" suffix (e.g., "SBIN.BO")
   - For US stocks: No suffix needed (e.g., "AAPL", "GOOGL", "TSLA")
   - Common Indian bank tickers: SBI/State Bank = "SBIN.NSE", HDFC Bank = "HDFCBANK.NSE", ICICI = "ICICIBANK.NSE"
   - For other markets: Use appropriate suffix (.L for London, .HK for Hong Kong)

2. **company_names**: Full company names corresponding to each symbol

3. **query_type**: One of: price, risk, sentiment, investment_decision, news_summary, comparison

4. **intent**: What the user wants to know

5. **time_frame**: Yahoo Finance period string: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max

6. **sentiment_focus**: positive, negative, neutral, mixed, or unknown

7. **news_category**: earnings, product, regulatory, merger, guidance, dividend, macro, analyst_ratings, legal, sector, general

Return ONLY this JSON format (no other text):
{{"yahoo_symbols": ["SBIN.NSE"], "company_names": ["State Bank of India"], "query_type": "investment_decision", "intent": "analyze SBI stock", "time_frame": "3mo", "sentiment_focus": "unknown", "news_category": "general"}}
"""
        try:
            response_payload = await self.backend.query_gemini(prompt)
            content = response_payload.get("response", "")
        except Exception as e:
            print(f"[GEMINI BACKEND ERROR] {str(e)}")
            response = self.llm.invoke(prompt)
            content = response.content
        import re
        import json
        
        try:
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(item.get('text', str(item)))
                    else:
                        text_parts.append(str(item))            
                content = ' '.join(text_parts)
            

            json_match = re.search(r'\{[^\{\}]*"(?:yahoo_symbols|company_names)"[^\{\}]*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                

                yahoo_symbols = parsed.get("yahoo_symbols", [])
                company_names = parsed.get("company_names", [])
                
                state["query_type"] = parsed.get("query_type", "investment_decision")
                state["intent"] = parsed.get("intent", query)
                state["sentiment_focus"] = self._normalize_sentiment_focus(
                    parsed.get("sentiment_focus")
                )
                state["news_category"] = self._normalize_news_category(
                    parsed.get("news_category")
                )
                raw_time_frame = parsed.get("time_frame")
                if raw_time_frame:
                    state["time_frame"] = self._normalize_timeframe(raw_time_frame)
                else:
                    state["time_frame"] = self._infer_timeframe_from_query(
                        query, state["intent"], state["sentiment_focus"]
                    )
                

                if yahoo_symbols:
                    resolved_symbols = []
                    symbol_metadata = {}
                    for i, symbol in enumerate(yahoo_symbols):
                        symbol = symbol.upper().strip()
                        if not any(suffix in symbol for suffix in ['.NSE','.BSE', '.BO', '.L', '.HK']) and len(symbol) <= 10:

                            pass 
                        name = company_names[i] if i < len(company_names) else symbol
                        resolved_symbols.append(symbol)
                        symbol_metadata[symbol] = {"name": name, "source": "gemini"}
                        print(f"[GEMINI] Resolved: {name} -> {symbol}")
                    
                    state["symbols"] = resolved_symbols
                    state["symbol_metadata"] = symbol_metadata
                

                elif company_names:
                    resolved_symbols = []
                    symbol_metadata = {}
                    for name in company_names:
                        print(f"[SEARCH] Resolving symbol for: {name}")
                        symbol, metadata = self.resolve_symbol(name)
                        if not symbol:
                            symbol, metadata = self.resolve_symbol(f"{name} India")
                        if symbol:
                            resolved_symbols.append(symbol)
                            if metadata:
                                symbol_metadata[symbol] = metadata
                            print(f"[RESOLVED] '{name}' -> {symbol}")
                        else:
                            print(f"[ERROR] Failed to resolve '{name}'")
                    
                    if resolved_symbols:
                        state["symbols"] = resolved_symbols
                        state["symbol_metadata"] = symbol_metadata
                    else:
                        print(f"[FALLBACK] No symbols found, trying to search entire query...")
                        symbol, metadata = self.resolve_symbol(query)
                        state["symbols"] = [symbol] if symbol else []
                        if symbol and metadata:
                            symbol_metadata[symbol] = metadata
                        state["symbol_metadata"] = symbol_metadata
                else:

                    print(f"[FALLBACK] No symbols in Gemini response, searching query...")
                    symbol, metadata = self.resolve_symbol(query)
                    state["symbols"] = [symbol] if symbol else []
                    if symbol and metadata:
                        state["symbol_metadata"] = {symbol: metadata}
                
            else:
                print(f"[FALLBACK] Could not parse JSON, searching query directly...")
                symbol, metadata = self.resolve_symbol(query)
                if symbol:
                    state["symbols"] = [symbol]
                    if metadata:
                        state["symbol_metadata"] = {symbol: metadata}
                else:
                    potential_tickers = re.findall(r'\b([A-Z]{2,10}(?:\.[A-Z]{1,2})?)\b', query.upper())
                    
                    # Filter out common English words
                    common_words = {
                        'THE', 'AND', 'FOR', 'WITH', 'THAT', 'THIS', 'FROM', 'HAVE', 'WILL',
                        'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHO', 'HOW', 'WHY', 'CAN', 'ALL',
                        'ANY', 'ARE', 'BUT', 'GET', 'HAS', 'HER', 'HIM', 'HIS', 'ITS', 'MAY',
                        'NEW', 'NOW', 'OLD', 'OUR', 'OUT', 'OWN', 'SAY', 'SHE', 'TOO', 'TWO',
                        'USE', 'WAY', 'ABOUT', 'AFTER', 'ALSO', 'BACK', 'BEEN', 'BEING',
                        'STOCK', 'PRICE', 'BUY', 'SELL', 'HOLD', 'MARKET', 'TRADE', 'INVEST',
                        'LAST', 'NEXT', 'GIVE', 'FINAL', 'PLAN', 'MONTHS', 'YEAR', 'WEEK',
                        'ANALYSIS', 'ANALYZE', 'RECOMMENDATION', 'SHOULD', 'COULD', 'WOULD',
                        'NEWS', 'SENTIMENT', 'RISK', 'RETURN', 'GROWTH', 'VALUE', 'TREND'
                    }
                    
                    valid_tickers = [t for t in potential_tickers if t not in common_words]
                    

                    if 1 <= len(valid_tickers) <= 3:
                        state["symbols"] = valid_tickers
                    else:
                        state["symbols"] = []
                
                state["query_type"] = "sentiment" if "sentiment" in query.lower() else "investment_decision"
                state["intent"] = query
                state["sentiment_focus"] = self._normalize_sentiment_focus(None)
                state["news_category"] = self._normalize_news_category(None)
                state["time_frame"] = self._infer_timeframe_from_query(
                    query, state["intent"], state["sentiment_focus"]
                )
                
        except Exception as e:
            print(f"[ERROR] Parse error: {e}")
            symbol, metadata = self.resolve_symbol(query)
            state["symbols"] = [symbol] if symbol else []
            if symbol and metadata:
                state["symbol_metadata"] = {symbol: metadata}
            state["query_type"] = "investment_decision"
            state["intent"] = query
            state["sentiment_focus"] = self._normalize_sentiment_focus(None)
            state["news_category"] = self._normalize_news_category(None)
            state["time_frame"] = self._infer_timeframe_from_query(
                query, state["intent"], state["sentiment_focus"]
            )
        
        print(f"\n[FINAL] Resolved symbols: {state['symbols']}")
        print(f"[FINAL] Query type: {state['query_type']}\n")
        
        state["step_count"]=1 
        return state

    def prefetch_context_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Prefetch core market data and news (TradingView + News)
        """
        symbols = state.get("symbols", [])
        if not symbols:
            print("[PREFETCH] No symbols to fetch data for")
            state["prefetch_results"] = {"error": "No valid stock symbol found in query"}
            state["step_count"] += 1
            return state
        symbol = symbols[0]

        prefetch_results = {}
        try:
            prefetch_results["tradingview"] = self.tradingview.get_current_price(symbol)
        except Exception as e:
            prefetch_results["tradingview_error"] = str(e)

        state["prefetch_results"] = prefetch_results
        state["step_count"] += 1
        return state

    def plan_analysis_node(self,state:AgentState)->AgentState:
        """
        Node 3: Prepare tools list (execute all tools)
        """
        symbols = state["symbols"]
        query_type = state.get("query_type", "investment_decision")
        intent = state.get("intent", "").lower()
        
        if len(symbols) > 1:
            state["tools_to_use"] = ["compare_stocks"]
        else:

            tools = [
                "get_stock_price",
                "get_stock_info"
            ]
            

            if query_type in ["sentiment", "news_summary"] or \
               any(word in intent for word in ["news", "sentiment", "headline", "article"]):
                tools.append("analyze_combined_news")
            
            state["tools_to_use"] = tools

        state["step_count"] += 1
        return state
    
    async def execute_tools_node(self, state: AgentState) -> AgentState:
        """Node 4: Execute tools"""
        tools_to_use = state["tools_to_use"]
        symbols = state["symbols"]
        if not symbols:
            print("[TOOLS] No symbols to analyze")
            state["tool_results"] = {"error": "No valid stock symbol found. Please specify a stock name or ticker."}
            state["step_count"] += 1
            return state
        results = {}
        prefetch = state.get("prefetch_results", {})
        timeframe = self._normalize_timeframe(state.get("time_frame"))
        category = self._normalize_news_category(state.get("news_category"))
        days = self._timeframe_to_days(timeframe)

        if "compare_stocks" in tools_to_use:
            try:
                tool = next((t for t in ALL_TOOLS if t.name == "compare_stocks"), None)
                if tool:
                    result = await tool.ainvoke({"symbols": symbols})
                    results["compare_stocks"] = result
            except Exception as e:
                results["compare_stocks"] = {"error": str(e)}

            if "get_market_news" in tools_to_use:
                try:
                    tool = next((t for t in ALL_TOOLS if t.name == "get_market_news"), None)
                    if tool:
                        results["get_market_news"] = await tool.ainvoke({"limit": 10})
                except Exception as e:
                    results["get_market_news"] = {"error": str(e)}
        else:
            symbol = symbols[0]
            company_name = self._get_company_name(
                symbol, state.get("symbol_metadata", {})
            )

            for tool_name in tools_to_use:
                try:
                    tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)
                    if not tool:
                        continue

                    if tool_name in ["get_stock_news", "analyze_news_sentiment", "summarize_news_articles"]:
                        result = await tool.ainvoke({
                            "symbol": symbol,
                            "company_name": company_name,
                            "days": days,
                            "category": category
                        })
                    elif tool_name == "analyze_combined_news":
                        result = await tool.ainvoke({
                            "symbol": symbol,
                            "company_name": company_name,
                            "days": days
                        })
                    elif tool_name in ["get_hist_data", "get_analyze_risk", "predict_price", "analyze_chart"]:
                        result = await tool.ainvoke({
                            "symbol": symbol,
                            "period": timeframe
                        })
                    elif tool_name == "get_market_news":
                        result = await tool.ainvoke({"limit": 10})
                    else:
                        result = await tool.ainvoke({"symbol": symbol})

                    results[tool_name] = result
                except Exception as e:
                    if tool_name == "get_stock_price" and prefetch.get("tradingview"):
                        results[tool_name] = json.dumps(prefetch["tradingview"], indent=2)
                    elif tool_name == "get_stock_news" and prefetch.get("news"):
                        results[tool_name] = json.dumps(prefetch["news"], indent=2)
                    else:
                        results[tool_name] = {"error": str(e)}

        state["tool_results"] = results
        state["step_count"] += 1
        return state
    
    def run_subagents_node(self, state: AgentState) -> AgentState:
        """
        Node 5: Run specialized sub-agent analyses
        """
        tool_results = state.get("tool_results", {})
        symbol = state["symbols"][0] if state.get("symbols") else "Unknown"
        timeframe = self._normalize_timeframe(state.get("time_frame"))
        sentiment_focus = self._normalize_sentiment_focus(state.get("sentiment_focus"))
        news_category = self._normalize_news_category(state.get("news_category"))

        parsed_results = {
            name: self._parse_tool_result(result)
            for name, result in tool_results.items()
        }

        hist_data = parsed_results.get("get_hist_data", [])
        if isinstance(hist_data, list) and len(hist_data) > 20:
            hist_data = hist_data[-20:]

        agent_inputs = {
            "market_data": {
                "price": parsed_results.get("get_stock_price", {}),
                "info": parsed_results.get("get_stock_info", {}),
                "metrics": parsed_results.get("get_financial_metrics", {})
            },
            "risk": {
                "risk": parsed_results.get("get_analyze_risk", {}),
                "market_maker_quote": parsed_results.get("get_market_maker_quote", {})
            },
            "sentiment": {
                "news": parsed_results.get("get_stock_news", []),
                "sentiment": parsed_results.get("analyze_news_sentiment", {}),
                "news_summary": parsed_results.get("summarize_news_articles", {}),
                "combined_news_analysis": parsed_results.get("analyze_combined_news", {})
            },
            "prediction": {
                "prediction": parsed_results.get("predict_price", {}),
                "chart": parsed_results.get("analyze_chart", {}),
                "recent_history": hist_data
            }
        }

        subagent_prompts = {
            "market_data": MARKET_DATA_AGENT_PROMPT,
            "risk": RISK_AGENT_PROMPT,
            "sentiment": SENTIMENT_AGENT_PROMPT,
            "prediction": PREDICTION_AGENT_PROMPT
        }

        reports = {}
        for agent_name, prompt in subagent_prompts.items():
            data_blob = json.dumps(agent_inputs.get(agent_name, {}), indent=2)
            agent_prompt = f"""{prompt.strip()}

Symbol: {symbol}
Time Frame: {timeframe}
Sentiment Focus: {sentiment_focus}
News Category: {news_category}

Data:
{data_blob}

Return concise bullet points (3-5) with actionable insights."""
            response = self.llm.invoke(agent_prompt)
            content = response.content
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            reports[agent_name] = content.strip()

        state["agent_reports"] = reports
        state["step_count"] += 1
        return state

    def synthesize_node(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize results"""
        import json
        tool_results = state["tool_results"]
        symbol = state["symbols"][0] if state["symbols"] else "Unknown"
        query_type = state["query_type"]
        timeframe = self._normalize_timeframe(state.get("time_frame"))
        sentiment_focus = self._normalize_sentiment_focus(state.get("sentiment_focus"))
        news_category = self._normalize_news_category(state.get("news_category"))
        agent_reports = state.get("agent_reports", {})
        parsed_results = {
            tool_name: self._parse_tool_result(result)
            for tool_name, result in tool_results.items()
        }
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
Time Frame: {timeframe}
Sentiment Focus: {sentiment_focus}
News Category: {news_category}

Financial Data:
{json.dumps(parsed_results, indent=2)}

Sub-Agent Reports:
{json.dumps(agent_reports, indent=2)}

News Articles (WITH LINKS):
{json.dumps(news_articles, indent=2)}

CRITICAL REQUIREMENTS:

1. **Key Insights** (3-5 bullet points with specific numbers from the data)
   - USE CORRECT CURRENCY SYMBOL based on stock exchange:
     * .NSE or .BO (India) → ₹
     * .L (London) → £
     * .PA, .DE, .AS, .MI, .MC (Europe) → €
     * .T (Tokyo) → ¥
     * .HK (Hong Kong) → HK$
     * No suffix or US exchanges → $

2. **Recommendation**: BUY, HOLD, or SELL with confidence level (high/medium/low)

3. **Detailed Reasoning** (2-3 sentences):
   - Reference SPECIFIC financial metrics (price, volatility, P/E ratio, etc.) with CORRECT CURRENCY
   - Mention sentiment analysis results if available
   - Cite news events if available

4. **Supporting News References** (ONLY IF NEWS PROVIDED):
   - If news_articles is empty, explicitly state "No news data available."
   - If provided, include 3-5 references in this format:
     • [Headline] - Source
       Link: [URL]
       Relevance: [1 sentence explaining how this supports your recommendation]

IMPORTANT:
- Always use the CORRECT currency symbol in all monetary values"""
        
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
        Node 7: Format final response with comprehensive data display
        """
        import json
        symbol=state["symbols"][0] if state["symbols"] else "Unknown"
        recommendation=state["recommendation"]  
        confidence=state["confidence"]
        full_analysis = state.get("full_analysis", "")
        timeframe = self._normalize_timeframe(state.get("time_frame"))
        sentiment_focus = self._normalize_sentiment_focus(state.get("sentiment_focus"))
        news_category = self._normalize_news_category(state.get("news_category"))
        tool_results = state.get("tool_results", {})
        news_refs = state.get("news_references", {})
        agent_reports = state.get("agent_reports", {})
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
        subagent_section = ""
        if agent_reports:
            title_map = {
                "market_data": "Market Data Agent",
                "risk": "Risk Agent",
                "sentiment": "Sentiment Agent",
                "prediction": "Prediction Agent"
            }
            subagent_section += "\n\n## 🧠 Sub-Agent Insights\n"
            for key in ["market_data", "risk", "sentiment", "prediction"]:
                report = agent_reports.get(key)
                if report:
                    subagent_section += f"\n### {title_map.get(key, key.title())}\n{report}\n"
        
        response=f"""📊 **Comprehensive Analysis for {symbol}**

**Recommendation:** {recommendation} (Confidence: {confidence.upper()})

---

## 🧭 Query Context

- **Time Frame:** {timeframe}
- **Sentiment Focus:** {sentiment_focus}
- **News Category:** {news_category}

---

## 🎯 Executive Summary

{full_analysis}

---

{subagent_section}

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
            "symbol_metadata":{},
            "query_type":"",
            "intent":"",
            "time_frame":"1mo",
            "sentiment_focus":"unknown",
            "news_category":"general",
            "tools_to_use":[],
            "tool_results":{},
            "prefetch_results":{},
            "market_data":{},
            "risk_analysis":{},
            "news_sentiment":{},
            "predictions":{},
            "insights":[],
            "recommendation":None,
            "confidence":"",
            "full_analysis":"",
            "news_references":{},
            "agent_reports":{},
            "should_continue":True,
            "error":None,
            "step_count":0            
        }
        final_state=await self.graph.ainvoke(initial_state)
        return final_state



