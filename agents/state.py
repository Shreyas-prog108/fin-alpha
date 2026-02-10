from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict,total=False):
    messages: List[Dict]
    user_query: str 
    
    # Query Analysis
    symbols: List[str]     
    symbol_metadata: Dict[str, Dict]
    query_type: str   
    intent: str                   
    time_frame: str
    sentiment_focus: str
    news_category: str
    
    # Tool Execution
    tools_to_use: List[str]          
    tool_results: Dict[str, Any]
    prefetch_results: Dict[str, Any]
    
    # Data Collection
    market_data: Dict                
    risk_analysis: Dict            
    news_sentiment: Dict         
    predictions: Dict           
    
    # Analysis
    insights: List[str]
    # Backward-compat alias used by older graph/state variants.
    sights: List[str]
    recommendation: Optional[str]    
    confidence: str
    full_analysis: str
    news_references: Dict
    agent_reports: Dict[str, str]
    
    # Metadata
    llm_call_count: int
    should_continue: bool            
    error: Optional[str]              
    step_count: int                  
