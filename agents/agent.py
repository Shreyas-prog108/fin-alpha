from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import Dict, List, Tuple, Optional
import json
import re

from .state import AgentState
from .config import config
from .tools import ALL_TOOLS
from .clients import get_alphavantage_client, get_backend_client
from .pdf_exporter import export_analysis_to_pdf
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

MAX_NEWS_ARTICLES_FOR_LLM = 3

class FinAgent:
    """
    Main Financial Analysis Agent using LangGraph
    """
    def __init__(self):
        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            api_key=config.GROQ_API_KEY,
        )
        self.graph=self._build_graph()
        self.alphavantage = get_alphavantage_client()
        self.backend = get_backend_client()

    async def _safe_generate(self, prompt: str, state: AgentState) -> str:
        """
        Quota-safe LLM generation wrapper.
        Enforces MAX_LLM_CALLS_PER_QUERY and prevents free-tier usage metrics.
        """
        current_usage = state.get("llm_call_count", 0)
        if current_usage >= config.MAX_LLM_CALLS_PER_QUERY:
             print(f"[QUOTA GUARD] Halting: Limit of {config.MAX_LLM_CALLS_PER_QUERY} calls reached.")
             return "Quota limit reached. stopping execution."

        state["llm_call_count"] = current_usage + 1
        print(f"[GROQ PAID TIER] Calling {config.GROQ_MODEL} (Call {state['llm_call_count']}/{config.MAX_LLM_CALLS_PER_QUERY})")
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            return str(content)
        except Exception as e:
             if "429" in str(e):
                 print(f"[QUOTA ERROR] 429 Hit. Stopping immediately to prevent free-tier penalty.")
                 return "Error: Rate limit hit."
             raise e

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

    async def _infer_timeframe_from_query(
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
            content = await self._safe_generate(prompt, {"llm_call_count": 0}) # Helper usage, not main flow
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
        return symbol.split(".", 1)[0]

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
        if exchange_upper in ["LSE", "LON", "XLON"]:
            return f"{ticker.upper()}.L"
        if exchange_upper in ["TYO", "JPX", "XTKS", "TSE"]:
            return f"{ticker.upper()}.T"
        if exchange_upper in ["HKG", "HKEX", "XHKG"]:
            return f"{ticker.upper()}.HK"
        if exchange_upper in ["SGX", "SES", "XSES"]:
            return f"{ticker.upper()}.SI"
        if exchange_upper in ["XETRA", "FRA", "XETR"]:
            return f"{ticker.upper()}.DE"
        if exchange_upper in ["EPA", "PAR", "XPAR"]:
            return f"{ticker.upper()}.PA"
        if exchange_upper in ["BME", "XMAD", "MAD"]:
            return f"{ticker.upper()}.MC"
        if exchange_upper in ["MIL", "XMIL"]:
            return f"{ticker.upper()}.MI"
        if exchange_upper in ["AMS", "XAMS"]:
            return f"{ticker.upper()}.AS"
        return ticker.upper()

    def _looks_like_ticker(self, value: str) -> bool:
        import re
        if not value:
            return False
        return bool(re.match(r"^[A-Z0-9&-]{1,15}(\.[A-Z]{1,4})?$", value.upper()))

    def _normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return ""
        try:
            return self.alphavantage.normalize_symbol(symbol.upper().strip())
        except Exception:
            return symbol.upper().strip()

    def _symbols_equivalent(self, left: str, right: str) -> bool:
        left_norm = self._normalize_symbol(left)
        right_norm = self._normalize_symbol(right)
        if not left_norm or not right_norm:
            return False
        if left_norm == right_norm:
            return True
        left_base, _, left_suffix = left_norm.partition(".")
        right_base, _, right_suffix = right_norm.partition(".")
        if left_base != right_base:
            return False
        if not left_suffix or not right_suffix:
            return True
        return left_suffix == right_suffix

    def _reconcile_symbol_with_company(
        self, symbol: str, company_name: str
    ) -> Tuple[str, Dict]:
        """Use Alpha Vantage search to verify/canonicalize LLM symbols."""
        normalized_input = self._normalize_symbol(symbol)
        fallback_meta = {
            "name": company_name or normalized_input,
            "source": "groq",
        }
        if not company_name:
            return normalized_input, fallback_meta

        try:
            av_match = self.alphavantage.search_symbol(company_name)
            match_score = float(av_match.get("match_score", 0) or 0) if av_match else 0.0
            if av_match and match_score >= 0.35:
                raw_symbol = (av_match.get("symbol") or av_match.get("ticker") or "").upper()
                exchange = av_match.get("exchange", "")
                candidate = (
                    raw_symbol
                    if "." in raw_symbol
                    else self._apply_exchange_suffix(av_match.get("ticker", ""), exchange)
                )
                candidate = self._normalize_symbol(candidate)
                if candidate and self._looks_like_ticker(candidate):
                    if normalized_input and not self._symbols_equivalent(normalized_input, candidate):
                        print(
                            f"[SYMBOL CORRECTED] {company_name}: "
                            f"{normalized_input} -> {candidate}"
                        )
                    return candidate, {
                        "name": av_match.get("name", company_name) or company_name,
                        "exchange": exchange,
                        "source": "alphavantage_verify",
                        "match_score": match_score,
                    }
        except Exception as e:
            print(f"[SYMBOL VERIFY ERROR] {str(e)}")

        return normalized_input, fallback_meta

    def _parse_tool_result(self, result: object) -> Dict:
        if isinstance(result, str):
            try:
                return json.loads(result)
            except Exception:
                return {"raw": result}
        if isinstance(result, dict):
            return result
        return {"raw": result}

    def _resolve_symbol_from_alphavantage(self, query: str) -> Tuple[Optional[str], Dict]:
        """Alpha Vantage symbol resolution."""
        try:
            result = self.alphavantage.search_symbol(query)
            if not result:
                return None, {}
            full_symbol = result.get("symbol", "")
            ticker = result.get("ticker", "")
            exchange = result.get("exchange", "")
            description = result.get("name", "") or query
            symbol = full_symbol.upper() if full_symbol and "." in full_symbol else self._apply_exchange_suffix(ticker, exchange)
            metadata = {
                "name": description,
                "exchange": exchange,
                "source": "alphavantage"
            }
            return symbol, metadata
        except Exception as e:
            print(f"[ALPHAVANTAGE ERROR] {str(e)}")
            return None, {}

    def _resolve_symbol_fallback(self, query: str) -> Tuple[Optional[str], Dict]:
        """
        Minimal fallback for symbol resolution.
        Only used when LLM doesn't provide symbols.
        Just formats the input - no hardcoded stock lists.
        """
        query_clean = query.strip()
        if not query_clean:
            return None, {}
        
        query_upper = query_clean.upper()
        if re.match(r"^[A-Z0-9&-]{1,15}\.[A-Z]{1,4}$", query_upper):
            suffix = query_upper.split(".")[-1]
            return query_upper, {"name": query_clean, "exchange": suffix, "source": "direct"}
        

        if len(query_upper) <= 15 and query_upper.replace("&", "").replace("-", "").replace(".", "").isalnum():
            return query_upper, {"name": query_clean, "exchange": "Unknown", "source": "fallback"}
        
        return None, {}

    def resolve_symbol(self, query: str) -> Tuple[Optional[str], Dict]:
        """
        Minimal symbol resolution fallback.
        LLM should handle all symbol resolution in parse_query_node.
        This is only used as a last resort.
        """
        query_clean = query.strip()
        if not query_clean:
            return None, {}
        
        symbol, metadata = self._resolve_symbol_fallback(query_clean)
        if symbol:
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
        Use Groq to extract stock symbols in exchange-qualified format.
        """   
        query=state["user_query"]
        prompt=f"""You are a financial assistant expert in stock markets. Analyze this query and extract information.

USER QUERY: "{query}"

Extract the following and return ONLY a valid JSON object:

1. **symbols**: Exact ticker symbols for the stocks mentioned.
   - Use exchange-qualified format where needed (e.g., SAP.DE, AIR.PA, 7203.T, D05.SI, RELIANCE.NSE).
   - For US symbols, no suffix needed when standard (e.g., AAPL, MSFT).
   - IMPORTANT: For non-US companies, use their PRIMARY/home exchange listing:
     * Indian stocks: use .NSE or .BSE (e.g., RELIANCE.NSE, TCS.NS)
     * European stocks: use locale.g., SAP exchange (.DE, SHEL.L)
     * Asian stocks: use local exchange (e.g., 7203.T, 9988.HK)
   - Avoid US OTC tickers (like GRSXY) when the company has a primary listing elsewhere

2. **company_names**: Full company names corresponding to each symbol

3. **query_type**: One of: price, risk, sentiment, investment_decision, news_summary, comparison

4. **intent**: What the user wants to know

5. **time_frame**: One of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max

6. **sentiment_focus**: positive, negative, neutral, mixed, or unknown

7. **news_category**: earnings, product, regulatory, merger, guidance, dividend, macro, analyst_ratings, legal, sector, general

Return ONLY this JSON format (no other text):
{{"symbols": ["SAP.DE"], "company_names": ["SAP SE"], "query_type": "investment_decision", "intent": "analyze SAP stock", "time_frame": "3mo", "sentiment_focus": "unknown", "news_category": "general"}}
"""
        try:
            response_payload = await self.backend.query_groq(prompt)
            content = response_payload.get("response", "")
        except Exception as e:
            print(f"[GROQ BACKEND ERROR] {str(e)}")
            content = await self._safe_generate(prompt, state)
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
            

            json_match = re.search(r'\{[^\{\}]*"(?:symbols|yahoo_symbols|company_names)"[^\{\}]*\}', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                

                symbols = parsed.get("symbols", []) or parsed.get("yahoo_symbols", [])
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
                    state["time_frame"] = await self._infer_timeframe_from_query(
                        query, state["intent"], state["sentiment_focus"]
                    )
                

                if symbols:
                    resolved_symbols = []
                    symbol_metadata = {}
                    for i, symbol in enumerate(symbols):
                        symbol = self._normalize_symbol(symbol)
                        name = company_names[i] if i < len(company_names) else symbol
                        reconciled_symbol, meta = self._reconcile_symbol_with_company(symbol, name)
                        if not reconciled_symbol:
                            continue
                        resolved_symbols.append(reconciled_symbol)
                        symbol_metadata[reconciled_symbol] = meta
                        print(f"[GROQ] Resolved: {name} -> {reconciled_symbol}")
                    
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

                    print(f"[FALLBACK] No symbols in LLM response, searching query...")
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
                    potential_tickers = re.findall(r'\b([A-Z0-9]{1,10}(?:\.[A-Z]{1,4})?)\b', query.upper())
                    
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
                state["time_frame"] = await self._infer_timeframe_from_query(
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
            state["time_frame"] = await self._infer_timeframe_from_query(
                query, state["intent"], state["sentiment_focus"]
            )
        
        print(f"\n[FINAL] Resolved symbols: {state['symbols']}")
        print(f"[FINAL] Query type: {state['query_type']}\n")
        
        state["step_count"]=1 
        return state

    def prefetch_context_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Prefetch core market data and news (Alpha Vantage + News)
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
            prefetch_results["alphavantage"] = self.alphavantage.get_current_price(symbol)
        except Exception as e:
            prefetch_results["alphavantage_error"] = str(e)

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
            # Core stack: quote/info + backend quantitative endpoints.
            tools = [
                "get_stock_price",
                "get_stock_info",
                "get_financial_metrics",
                "get_hist_data",
                "get_analyze_risk",
                "predict_price",
                "analyze_chart",
            ]

            # Always include base sentiment/news context.
            tools.extend([
                "get_stock_news",
                "analyze_news_sentiment",
                "analyze_combined_news",
            ])

            # For news-heavy queries, add backend news synthesis endpoints.
            if query_type in ["sentiment", "news_summary"] or \
               any(word in intent for word in ["news", "sentiment", "headline", "article"]):
                tools.extend([
                    "summarize_news_articles",
                ])

            if query_type == "risk" or any(word in intent for word in ["risk", "volatility", "spread"]):
                tools.append("get_market_maker_quote")

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
                    if tool_name == "get_stock_price" and prefetch.get("alphavantage"):
                        results[tool_name] = json.dumps(prefetch["alphavantage"], indent=2)
                    elif tool_name == "get_stock_news" and prefetch.get("news"):
                        results[tool_name] = json.dumps(prefetch["news"], indent=2)
                    else:
                        results[tool_name] = {"error": str(e)}

        state["tool_results"] = results
        state["step_count"] += 1
        return state
    
    async def run_subagents_node(self, state: AgentState) -> AgentState:
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
        stock_news_payload = parsed_results.get("get_stock_news", {})
        stock_news_articles = []
        if isinstance(stock_news_payload, dict):
            stock_news_articles = stock_news_payload.get("articles", [])
        elif isinstance(stock_news_payload, list):
            stock_news_articles = stock_news_payload

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
                "news": stock_news_articles,
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
             # Quota guard check before preparation
            if state.get("llm_call_count", 0) >= config.MAX_LLM_CALLS_PER_QUERY:
                break
            data_blob = json.dumps(agent_inputs.get(agent_name, {}), indent=2)
            agent_prompt = f"""{prompt.strip()}

SYMBOL: {symbol}
TIME FRAME: {timeframe}
SENTIMENT FOCUS: {sentiment_focus}
NEWS CATEGORY: {news_category}

DATA INPUT:
{data_blob}

IMPORTANT: Follow the MANDATORY OUTPUT FORMAT specified in your instructions above. Provide detailed analysis with specific numbers. Do NOT provide BUY/SELL recommendations - focus on your specialized analysis."""
            content = await self._safe_generate(agent_prompt, state)
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            reports[agent_name] = content.strip()

        state["agent_reports"] = reports
        state["step_count"] += 1
        return state

    async def synthesize_node(self, state: AgentState) -> AgentState:
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
        combined_news_data = parsed_results.get("analyze_combined_news", {})

        stock_news_articles = []
        if isinstance(news_data, dict):
            stock_news_articles = news_data.get("articles", [])
        elif isinstance(news_data, list):
            stock_news_articles = news_data

        news_articles = []
        for article in stock_news_articles[:MAX_NEWS_ARTICLES_FOR_LLM]:
            if isinstance(article, dict):
                news_articles.append({
                    "title": article.get("title", "N/A"),
                    "url": article.get("url", ""),
                    "source": article.get("source", "Unknown"),
                    "sentiment": article.get("sentiment", "neutral"),
                    "published_at": article.get("published_at", article.get("publishedAt", "")),
                })

        combined_headlines = []
        if isinstance(combined_news_data, dict):
            raw_headlines = combined_news_data.get("top_headlines", [])
            if isinstance(raw_headlines, list):
                for item in raw_headlines[:MAX_NEWS_ARTICLES_FOR_LLM]:
                    if isinstance(item, dict):
                        combined_headlines.append(item)
                    else:
                        text = str(item)
                        clean = text.strip().lower()
                        if clean.startswith("[mint]"):
                            source = "LiveMint"
                        elif clean.startswith("[newsapi]"):
                            source = "NewsAPI"
                        elif clean.startswith("[perplexity]"):
                            source = "Perplexity"
                        else:
                            source = "Unknown"
                        title = text.split("]", 1)[-1].strip() if text.startswith("[") and "]" in text else text
                        combined_headlines.append({
                            "title": title,
                            "source": source,
                            "url": "",
                            "sentiment": "neutral",
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
   - USE CORRECT CURRENCY SYMBOL:
     * First preference: infer from fetched data fields like `currency` in price/info.
     * If currency is unavailable, use exchange suffix heuristics (e.g., .NSE/.BO → ₹, .L → £, .T → ¥, .HK → HK$, .SI → S$, .SW → CHF, many Europe exchanges → €).
     * If still unclear, use the ISO currency code from data instead of guessing.

2. **Recommendation**: BUY, HOLD, or SELL with confidence level (high/medium/low)

3. **Detailed Reasoning** (2-3 sentences):
   - Reference SPECIFIC financial metrics (price, volatility, P/E ratio, etc.) with CORRECT CURRENCY
   - Mention sentiment analysis results if available
   - Cite news events if available

4. **Supporting News References** (ONLY IF NEWS PROVIDED):
   - If news_articles is empty, explicitly state "No news data available."
   - If provided, include 1-3 references in this format:
     • [Headline] - Source
       Link: [URL]
       Relevance: [1 sentence explaining how this supports your recommendation]

IMPORTANT:
- Always use the CORRECT currency symbol in all monetary values"""
        
        content = await self._safe_generate(prompt, state)
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
        headlines = stock_news_articles[:MAX_NEWS_ARTICLES_FOR_LLM] if isinstance(stock_news_articles, list) else []
        sentiment = "neutral"
        sentiment_score = 0.0
        if isinstance(sentiment_data, dict):
            sentiment = sentiment_data.get("overall_sentiment", sentiment_data.get("sentiment", "neutral"))
            sentiment_score = sentiment_data.get("sentiment_score", sentiment_data.get("average_score", 0.0))
        
        state["news_references"] = {
            "headlines": headlines[:3] if isinstance(headlines, list) else [], 
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "combined_headlines": combined_headlines[:MAX_NEWS_ARTICLES_FOR_LLM] if isinstance(combined_headlines, list) else [],
            "combined_analysis": combined_news_data.get("analysis", "") if isinstance(combined_news_data, dict) else "",
            "combined_sentiment": combined_news_data.get("sentiment_summary", "neutral") if isinstance(combined_news_data, dict) else "neutral",
            "combined_articles_analyzed": combined_news_data.get("articles_analyzed", 0) if isinstance(combined_news_data, dict) else 0,
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
            current_price = price_data.get('current_price', 0)
            market_cap = price_data.get('market_cap', 0)
            pe_ratio = price_data.get('pe_ratio', 0)
            currency = price_data.get('currency', '$')
            
            # Also try to get from financial metrics if not in price data
            metrics_data = tool_results.get("get_financial_metrics", {})
            if isinstance(metrics_data, str):
                try:
                    metrics_data = json.loads(metrics_data)
                except:
                    metrics_data = {}
            
            # Use financial metrics as fallback for missing data
            if (not market_cap or market_cap == 0) and metrics_data:
                market_cap = metrics_data.get('market_cap', 0)
            if (not pe_ratio or pe_ratio == 0) and metrics_data:
                pe_ratio = metrics_data.get('pe_ratio', 0)
            
            # Format market cap nicely (handle Indian stock format - crores)
            if market_cap and market_cap > 0:
                if currency == 'INR':
                    # Indian format - crores
                    if market_cap >= 1e7:  # 10 crore
                        market_cap_str = f"₹{market_cap/1e7:.2f}Cr"
                    else:
                        market_cap_str = f"₹{market_cap:,.0f}"
                else:
                    # US format
                    if market_cap >= 1e12:
                        market_cap_str = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"${market_cap/1e9:.2f}B"
                    else:
                        market_cap_str = f"${market_cap:,.0f}"
            else:
                market_cap_str = "N/A"
            
            # Format PE ratio
            if pe_ratio and pe_ratio > 0:
                pe_str = f"{pe_ratio:.2f}x"
            else:
                pe_str = "N/A"
            
            # Format price
            if current_price and current_price > 0:
                price_str = f"{currency}{current_price:.2f}"
            else:
                price_str = "N/A"
            
            # Get 52-week data from price or metrics
            week52_low = price_data.get('fifty_two_week_low') or (metrics_data.get('fifty_two_week_low') if metrics_data else None)
            week52_high = price_data.get('fifty_two_week_high') or (metrics_data.get('fifty_two_week_high') if metrics_data else None)
            sector = price_data.get('sector') or (metrics_data.get('sector') if metrics_data else None) or "N/A"
            industry = price_data.get('industry') or (metrics_data.get('industry') if metrics_data else None) or "N/A"
            
            detailed_sections.append(f"""
### 📈 Current Market Data

- **Current Price:** {price_str}
- **Market Cap:** {market_cap_str}
- **P/E Ratio:** {pe_str}
- **Day Change:** {price_data.get('change', 0):+.2f} ({price_data.get('change_percent', 0):+.2f}%)
- **Day Range:** {currency}{price_data.get('low', 'N/A')} - {currency}{price_data.get('high', 'N/A')}
- **52-Week Range:** {currency}{week52_low if week52_low else 'N/A'} - {currency}{week52_high if week52_high else 'N/A'}
- **Volume:** {price_data.get('volume', 0):,.0f}
- **Exchange:** {price_data.get('exchange', 'N/A')}
- **Sector:** {sector}
- **Industry:** {industry}
""")
        metrics_data = tool_results.get("get_financial_metrics", {})
        if isinstance(metrics_data, str):
            try:
                metrics_data = json.loads(metrics_data)
            except:
                metrics_data = {}
        
        # Helper to format ratio values
        def fmt_ratio(value, suffix='x'):
            if value and value > 0:
                return f"{value:.2f}{suffix}"
            return "N/A"
        
        def fmt_pct(value):
            if value and value != 0:
                return f"{value:+.2f}%"
            return "N/A"
        
        def fmt_number(value):
            if value and value != 0:
                return f"{value:,.2f}"
            return "N/A"
        
        if metrics_data and not metrics_data.get("error"):
            detailed_sections.append(f"""
### 💰 Financial Metrics

**Valuation Ratios:**
- P/E Ratio: {fmt_ratio(metrics_data.get('pe_ratio'))}
- Forward P/E: {fmt_ratio(metrics_data.get('forward_pe'))}
- Price to Book (P/B): {fmt_ratio(metrics_data.get('price_to_book'))}
- Price to Sales (P/S): {fmt_ratio(metrics_data.get('price_to_sales'))}
- EV/EBITDA: {fmt_ratio(metrics_data.get('ev_to_ebitda'))}
- PEG Ratio: {fmt_ratio(metrics_data.get('peg_ratio'))}
- Dividend Yield: {fmt_pct(metrics_data.get('dividend_yield'))}

**Profitability:**
- Profit Margin: {fmt_pct(metrics_data.get('profit_margin'))}
- Operating Margin: {fmt_pct(metrics_data.get('operating_margin'))}
- Gross Margin: {fmt_pct(metrics_data.get('gross_margin'))}
- ROE (Return on Equity): {fmt_pct(metrics_data.get('roe'))}
- ROA (Return on Assets): {fmt_pct(metrics_data.get('roa'))}
- ROIC (Return on Invested Capital): {fmt_pct(metrics_data.get('roic'))}

**Growth:**
- Revenue Growth (YoY): {fmt_pct(metrics_data.get('revenue_growth'))}
- Earnings Growth (YoY): {fmt_pct(metrics_data.get('earnings_growth'))}
- EPS (Earnings Per Share): {fmt_number(metrics_data.get('eps'))}

**Financial Health:**
- Debt to Equity: {fmt_ratio(metrics_data.get('debt_to_equity'))}
- Current Ratio: {fmt_ratio(metrics_data.get('current_ratio'))}
- Quick Ratio: {fmt_ratio(metrics_data.get('quick_ratio'))}
- Net Debt/EBITDA: {fmt_ratio(metrics_data.get('net_debt_to_ebitda'))}

**Company Info:**
- Market Cap: {fmt_number(metrics_data.get('market_cap'))}
- Enterprise Value: {fmt_number(metrics_data.get('enterprise_value'))}
- Sector: {metrics_data.get('sector', 'N/A')}
- Industry: {metrics_data.get('industry', 'N/A')}
- Beta: {fmt_ratio(metrics_data.get('beta', 0), '')}
""")
        risk_data = tool_results.get("get_analyze_risk", {})
        if isinstance(risk_data, str):
            try:
                risk_data = json.loads(risk_data)
            except:
                risk_data = {}
        
        # Helper for risk metrics
        def fmt_pct_risk(value):
            if value and value != 0:
                return f"{value*100:.2f}%"
            return "N/A"
        
        def fmt_ratio_risk(value):
            if value and value != 0:
                return f"{value:.2f}"
            return "N/A"
        
        if risk_data and not risk_data.get("error"):
            detailed_sections.append(f"""
### ⚠️ Risk Analysis

**Volatility Metrics:**
- Annualized Volatility: {fmt_pct_risk(risk_data.get('volatility', 0))}
- Daily Volatility: {fmt_pct_risk(risk_data.get('daily_volatility', 0))}
- Volatility vs Sector: {risk_data.get('volatility_vs_sector', 'N/A')}

**Downside Risk:**
- Maximum Drawdown: {fmt_pct_risk(risk_data.get('max_drawdown', 0))}
- Value at Risk (VaR 95%): {fmt_ratio_risk(risk_data.get('var_95', 0))}
- Conditional VaR (CVaR 95%): {fmt_ratio_risk(risk_data.get('cvar_95', 0))}
- Sortino Ratio: {fmt_ratio_risk(risk_data.get('sortino_ratio', 0))}

**Risk-Adjusted Returns:**
- Sharpe Ratio: {fmt_ratio_risk(risk_data.get('sharpe_ratio', 0))}
- Treynor Ratio: {fmt_ratio_risk(risk_data.get('treynor_ratio', 0))}
- Information Ratio: {fmt_ratio_risk(risk_data.get('information_ratio', 0))}

**Market Sensitivity:**
- Beta: {fmt_ratio_risk(risk_data.get('beta', 0))}
- Alpha: {fmt_ratio_risk(risk_data.get('alpha', 0))}
- R-Squared: {fmt_ratio_risk(risk_data.get('r_squared', 0))}

**Risk Assessment:**
- Overall Risk Level: {risk_data.get('risk_level', 'N/A').upper()}
- Risk Flag: {risk_data.get('risk_flag', 'N/A')}
""")
        prediction_data = tool_results.get("predict_price", {})
        if isinstance(prediction_data, str):
            try:
                prediction_data = json.loads(prediction_data)
            except:
                prediction_data = {}
        
        if prediction_data and not prediction_data.get("error"):
            predicted_price = prediction_data.get('predicted_price', 0)
            current_price_pred = prediction_data.get('current_price', 0)
            currency_pred = prediction_data.get('currency', '$')
            
            if predicted_price and predicted_price > 0:
                pred_price_str = f"{currency_pred}{predicted_price:.2f}"
            else:
                pred_price_str = "N/A"
            
            if current_price_pred and current_price_pred > 0:
                curr_price_str = f"{currency_pred}{current_price_pred:.2f}"
            else:
                curr_price_str = "N/A"
            
            detailed_sections.append(f"""
### 🔮 Price Prediction

**Price Outlook:**
- Current Price: {curr_price_str}
- Predicted Price ({prediction_data.get('timeframe', 'Next Period')}): {pred_price_str}
- Expected Change: {prediction_data.get('predicted_change_percent', 0):+.2f}%

**Model Details:**
- Method: {prediction_data.get('method', 'N/A').upper()}
- Confidence: {prediction_data.get('confidence', 'N/A')}
- Model Accuracy: {prediction_data.get('accuracy', 'N/A')}

**Price Scenarios:**
- Bull Case: {prediction_data.get('bull_case', 'N/A')}
- Bear Case: {prediction_data.get('bear_case', 'N/A')}
- Base Case: {prediction_data.get('base_case', 'N/A')}

**Key Levels:**
- Support: {prediction_data.get('support_level', 'N/A')}
- Resistance: {prediction_data.get('resistance_level', 'N/A')}
""")
        sentiment_data = tool_results.get("analyze_news_sentiment", {})
        if isinstance(sentiment_data, str):
            try:
                sentiment_data = json.loads(sentiment_data)
            except:
                sentiment_data = {}
        
        if sentiment_data and not sentiment_data.get("error"):
            total_articles = sentiment_data.get('total_articles', 0)
            positive = sentiment_data.get('positive_count', 0)
            negative = sentiment_data.get('negative_count', 0)
            neutral = sentiment_data.get('neutral_count', 0)
            
            # Calculate percentages
            if total_articles > 0:
                pos_pct = (positive / total_articles) * 100
                neg_pct = (negative / total_articles) * 100
                neu_pct = (neutral / total_articles) * 100
            else:
                pos_pct = neg_pct = neu_pct = 0
            
            detailed_sections.append(f"""
### 📊 News Sentiment Analysis

**Sentiment Overview:**
- Overall Sentiment: {sentiment_data.get('overall_sentiment', 'N/A').upper()}
- Sentiment Score: {sentiment_data.get('sentiment_score', 0):+.2f} (-1 to +1 scale)
- Confidence: {sentiment_data.get('confidence', 'N/A').upper()}

**Article Breakdown ({total_articles} articles):**
- Positive: {positive} articles ({pos_pct:.1f}%)
- Negative: {negative} articles ({neg_pct:.1f}%)
- Neutral: {neutral} articles ({neu_pct:.1f}%)

**Summary:** {sentiment_data.get('summary', 'N/A')}

**Top Headlines:**
{chr(10).join(f"- {title}" for title in sentiment_data.get('top_headlines', [])[:5])}
""")
        combined_news_data = tool_results.get("analyze_combined_news", {})
        if isinstance(combined_news_data, str):
            try:
                combined_news_data = json.loads(combined_news_data)
            except:
                combined_news_data = {}

        if combined_news_data and not combined_news_data.get("error"):
            detailed_sections.append(f"""
### 🗞️ Multi-Source News Analysis (Perplexity + NewsAPI + LiveMint)

- **Combined Sentiment:** {combined_news_data.get('sentiment_summary', 'N/A').upper()}
- **Articles Analyzed:** {combined_news_data.get('articles_analyzed', 0)}
- **Analysis:** {combined_news_data.get('analysis', 'N/A')}
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
        
        # Market Maker Quote
        market_maker_data = tool_results.get("get_market_maker_quote", {})
        if isinstance(market_maker_data, str):
            try:
                market_maker_data = json.loads(market_maker_data)
            except:
                market_maker_data = {}
        
        if market_maker_data and not market_maker_data.get("error"):
            mm_mid = market_maker_data.get('mid_price', 0)
            mm_spread = market_maker_data.get('spread', 0)
            mm_spread_pct = market_maker_data.get('spread_percent', 0)
            mm_currency = '$'
            
            bid_val = market_maker_data.get('bid_price', market_maker_data.get('bid', 'N/A'))
            ask_val = market_maker_data.get('ask_price', market_maker_data.get('ask', 'N/A'))
            res_price = market_maker_data.get('reservation_price', 0)
            vol_used = market_maker_data.get('volatility_used', 0)
            inv_val = market_maker_data.get('params', {}).get('inventory', 0)
            time_horizon = market_maker_data.get('params', {}).get('T', 'N/A')
            
            detailed_sections.append(f"""
### 📊 Market Maker Quote (Avellaneda-Stoikov)

**Spread Analysis:**
- Optimal Bid Price: {mm_currency}{bid_val}
- Optimal Ask Price: {mm_currency}{ask_val}
- Spread: {mm_currency}{mm_spread:.4f} ({mm_spread_pct:.4f}%)

**Parameters Used:**
- Mid Price: {mm_currency}{mm_mid:.2f}
- Volatility (Annualized): {vol_used*100:.2f}%
- Risk Aversion (γ): {market_maker_data.get('risk_aversion', 'N/A')}
- Order Intensity (κ): {market_maker_data.get('kappa', 'N/A')}
- Time Horizon: {time_horizon} years

**Inventory Risk:**
- Current Inventory: {inv_val:.2f} units
- Reservation Price: {mm_currency}{res_price:.2f}
- Adjustment: {market_maker_data.get('inventory_adjustment', 'N/A')}
- Quote Recommendation: {market_maker_data.get('quote_recommendation', 'N/A')}
""")
        news_section = ""  
        if news_refs:
            headlines = news_refs.get("headlines", [])
            sentiment = news_refs.get("sentiment", "neutral")
            score = news_refs.get("sentiment_score", 0)
            combined_headlines = news_refs.get("combined_headlines", [])
            combined_sentiment = news_refs.get("combined_sentiment", "neutral")
            combined_count = news_refs.get("combined_articles_analyzed", 0)
            combined_analysis = news_refs.get("combined_analysis", "")
            
            if headlines and len(headlines) > 0:
                news_section = f"\n\n### 📰 Recent News Articles\n\n"
                news_section += f"**Market Sentiment:** {sentiment.upper()} (Score: {score:.2f})\n\n"
                for i, headline in enumerate(headlines[:MAX_NEWS_ARTICLES_FOR_LLM], 1):
                    if isinstance(headline, dict):
                        title = headline.get('title', headline.get('headline', headline.get('description', 'N/A')))
                        source = headline.get('source', headline.get('publisher', headline.get('author', 'Unknown')))
                        url = headline.get('url', headline.get('link', '#'))
                        published = headline.get('published_at', headline.get('publishedAt', headline.get('date', '')))
                        
                        news_section += f"\n**{i}. {title}**\n"
                        news_section += f"   - Source: {source}"
                        if published:
                            news_section += f" | Date: {published[:10]}"
                        news_section += "\n"
                        if url and url != '#':
                            news_section += f"   - Link: {url}\n"
                    else:
                        news_section += f"{i}. {headline}\n"
            if combined_headlines:
                news_section += "\n\n### 🗞️ Combined News (Includes LiveMint)\n\n"
                news_section += (
                    f"**Combined Sentiment:** {combined_sentiment.upper()} "
                    f"(Articles: {combined_count})\n\n"
                )
                if combined_analysis:
                    news_section += f"{combined_analysis}\n\n"
                for i, headline in enumerate(combined_headlines[:MAX_NEWS_ARTICLES_FOR_LLM], 1):
                    if isinstance(headline, dict):
                        title = headline.get("title", "N/A")
                        source = headline.get("source", "Unknown")
                        url = headline.get("url", "")
                        news_section += f"{i}. {title} - {source}\n"
                        if url:
                            news_section += f"   Link: {url}\n"
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

        try:
            pdf_path = export_analysis_to_pdf(symbol=symbol, response_text=response)
            response = f"{response}\n\nPDF Export: `{pdf_path}`"
        except Exception as pdf_error:
            response = f"{response}\n\nPDF Export: failed ({str(pdf_error)})"
        
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
            "llm_call_count":0,
            "should_continue":True,
            "error":None,
            "step_count":0            
        }
        final_state=await self.graph.ainvoke(initial_state)
        return final_state
