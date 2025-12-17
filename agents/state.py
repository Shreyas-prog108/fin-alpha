from typing import TypedDict, List, Dict, Optional, Literal

class AgentState(TypedDict):
    messages: List[Dict]
    user_query: str 
    
    # Query Analysis
    symbols: List[str]     
    query_type: str   
    intent: str                   
    
    # Tool Execution
    tools_to_use: List[str]          
    tool_results: Dict[str, any]    
    
    # Data Collection
    market_data: Dict                
    risk_analysis: Dict            
    news_sentiment: Dict         
    predictions: Dict           
    
    # Analysis
    insights: List[str]              
    recommendation: Optional[str]    
    confidence: str                
    
    # Metadata
    should_continue: bool            
    error: Optional[str]              
    step_count: int                  