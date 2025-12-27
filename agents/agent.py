from langgraph.graph import StateGraph,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict
import os
import json

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
        self.llm_with_tools=self.llm.bind_tools(ALL_TOOLS)
        self.graph=self._build_graph()

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
        Extract symbols,determine intent
        """   
        query=state["user_query"]
        prompt=f"""You are a financial assistant. Extract stock ticker symbols from this query:

"{query}"

Instructions:
- Convert company names to ticker symbols:
  * "State Bank of India" or "SBI" -> "SBIN.NS"
  * "Reliance" -> "RELIANCE.NS"
  * "Apple" -> "AAPL"
- For Indian stocks, ALWAYS use .NS suffix (NSE exchange)
- Determine query type: price, risk, sentiment, investment_decision, news

Return ONLY valid JSON (no markdown, no extra text):
{{"symbols":["SBIN.NS"],"query_type":"investment_decision","intent":"2 year investment analysis"}}
"""
        response=self.llm.invoke(prompt)
        import re
        import json
        
        # Try to parse LLM response as JSON
        try:
            content = response.content
            
            # Handle if content is a list (some LLM responses)
            if isinstance(content, list):
                # Extract text from list of dicts
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        text_parts.append(item.get('text', str(item)))
                    else:
                        text_parts.append(str(item))
                content = ' '.join(text_parts)
            
            print(f"\n[DEBUG] LLM Response: {content[:200]}...")  # Debug output
            
            # Try to extract JSON - look for curly braces
            json_match = re.search(r'\{[^\{\}]*"symbols"[^\{\}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                state["symbols"] = parsed.get("symbols", ["AAPL"])
                state["query_type"] = parsed.get("query_type", "investment_decision")
                state["intent"] = parsed.get("intent", query)
                print(f"[DEBUG] Parsed symbols: {state['symbols']}")  # Debug output
            else:
                print("[DEBUG] No JSON found, using fallback...")  # Debug output
                # Fallback: check for common Indian company names
                query_lower = query.lower()
                if "state bank" in query_lower or "sbi" in query_lower:
                    state["symbols"] = ["SBIN.NS"]
                elif "reliance" in query_lower:
                    state["symbols"] = ["RELIANCE.NS"]
                elif "tcs" in query_lower or "tata consultancy" in query_lower:
                    state["symbols"] = ["TCS.NS"]
                elif "infosys" in query_lower:
                    state["symbols"] = ["INFY.NS"]
                else:
                    # Try to extract ticker symbols from query
                    symbols = re.findall(r'\b[A-Z]{2,5}(?:\.(?:NS|BO))?\b', query.upper())
                    state["symbols"] = symbols if symbols else ["AAPL"]
                
                state["query_type"] = "sentiment" if "sentiment" in query.lower() else "investment_decision"
                state["intent"] = query
        except Exception as e:
            print(f"[DEBUG] Error parsing: {e}")  # Debug output
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
        if query_type=="price":
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
        symbol = state["symbols"][0] if state["symbols"] else "AAPL"
        results = {}
        
        print(f"\n[DEBUG] Executing {len(tools_to_use)} tools for {symbol}...")  # Debug output
        
        # Extract company name from symbol (remove .NS, .BO suffixes)
        company_name = symbol.replace('.NS', '').replace('.BO', '')
        
        # Map symbol to full company name for better news results
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
                    print(f"[DEBUG] Running tool: {tool_name}")
                    if tool_name in ["get_stock_news", "analyze_news_sentiment"]:
                        result = await tool.ainvoke({
                            "symbol": symbol,
                            "company_name": full_company_name
                        })
                    else:
                        result = await tool.ainvoke({"symbol": symbol})
                    
                    results[tool_name] = result
                    print(f"[DEBUG] Tool {tool_name} completed")
                else:
                    print(f"[DEBUG] Tool {tool_name} not found in ALL_TOOLS")
            except Exception as e:
                print(f"[DEBUG] Tool {tool_name} error: {e}")
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
        
        print(f"\n[DEBUG] News data keys: {list(news_data.keys()) if isinstance(news_data, dict) else 'Not a dict'}")
        if isinstance(news_data, dict) and 'error' in news_data:
            print(f"[DEBUG] News API error: {news_data['error']}")
        print(f"[DEBUG] Sentiment data keys: {list(sentiment_data.keys()) if isinstance(sentiment_data, dict) else 'Not a dict'}")
        if isinstance(sentiment_data, dict) and 'error' in sentiment_data:
            print(f"[DEBUG] Sentiment API error: {sentiment_data['error']}")
        
        prompt = f"""You are a financial analyst. Analyze the following data for {symbol}:

Query Type: {query_type}
Tool Results:
{json.dumps(tool_results, indent=2)}

IMPORTANT: Your analysis MUST include:
1. 3-5 key insights (bullet points)
2. Clear recommendation: BUY, HOLD, or SELL
3. Confidence level: high, medium, or low
4. Detailed reasoning (2-3 sentences) that EXPLICITLY references specific news headlines or events from the news data
5. If news data is available, cite specific headlines that support your recommendation

Format your response with clear sections and include news references in your reasoning."""
        
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
        
        # Extract insights - look for bullet points or numbered items
        insights = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line[0:2].replace('.','').isdigit()):
                insights.append(line.lstrip('-•*0123456789. '))
        
        state["insights"] = insights if insights else [content[:200] + "..."]
        state["full_analysis"] = content
        
        # Store news references separately for better formatting
        # Handle different possible structures from news API
        headlines = []
        if isinstance(news_data, dict):
            headlines = news_data.get("headlines", news_data.get("articles", news_data.get("news", [])))
        
        # Extract sentiment info
        sentiment = "neutral"
        sentiment_score = 0.0
        if isinstance(sentiment_data, dict):
            sentiment = sentiment_data.get("overall_sentiment", sentiment_data.get("sentiment", "neutral"))
            sentiment_score = sentiment_data.get("score", sentiment_data.get("sentiment_score", 0.0))
        
        print(f"[DEBUG] Extracted {len(headlines) if isinstance(headlines, list) else 0} headlines")
        
        state["news_references"] = {
            "headlines": headlines[:3] if isinstance(headlines, list) else [],  # Top 3 only
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
        
        # Format news references section
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
            "should_continue":True,
            "error":None,
            "step_count":0            
        }
        final_state=await self.graph.ainvoke(initial_state)
        return final_state



