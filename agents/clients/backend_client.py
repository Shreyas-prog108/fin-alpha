import httpx
import os
from typing import Dict, List, Optional

class BackendClient:
    """
    Client for FinAgent FastAPI backend
    Calls quantitative analysis endpoints
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("API_KEY")
        self.timeout = 30.0
        self._client = None
        
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("BACKEND_URL must start with http:// or https://")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of async client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
    
    def _get_headers(self) -> Dict:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def health_check(self) -> Dict:
        """
        Check if backend is running
        
        Returns:
            Dictionary with service status
        
        Raises:
            Exception: If backend is unreachable
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/health",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Backend health check failed: {str(e)}")
    
    async def analyze_risk(self, symbol: str, data: List[Dict]) -> Dict:
        """
        Analyze stock risk using backend API
        
        Args:
            symbol: Stock ticker symbol
            data: Historical price data with format:
                  [{"time": "2025-01-01", "close": 150.0, "volume": 50000}, ...]
        
        Returns:
            Dictionary with risk analysis:
            {
                "symbol": "AAPL",
                "volatility": 0.25,
                "risk_level": "medium",
                "avg_volume": 75000000,
                "volume_std": 25000000,
                "anomalies": [...]
            }
        
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "symbol": symbol,
                "data": data
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/analyze-risk",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise Exception(f"Risk analysis API error: {str(e)}")
    
    async def predict_price(
        self, 
        symbol: str, 
        data: List[Dict],
        method: str = "ema",
        ema_span: int = 5
    ) -> Dict:
        """
        Predict stock price using backend API
        
        Args:
            symbol: Stock ticker symbol
            data: Historical price data with format:
                  [{"close": 150.0}, {"close": 148.0}, ...]
            method: Prediction method - "ema" or "linear"
            ema_span: EMA span for EMA method
        
        Returns:
            Dictionary with prediction:
            {
                "symbol": "AAPL",
                "method": "ema",
                "current_price": 185.92,
                "predicted_price": 189.45,
                "change": 3.53,
                "change_percent": 1.9,
                "confidence": "medium",
                "trend": "bullish"
            }
        
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "symbol": symbol,
                "method": method,
                "data": data,
                "ema_span": ema_span
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/predict-price",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise Exception(f"Price prediction API error: {str(e)}")
    
    async def get_market_maker_quote(
        self,
        mid_price: float,
        volatility: float,
        risk_aversion: float = 0.1,
        time_horizon: float = 1.0,
        inventory: float = 0.0,
        kappa: float = 1.5
    ) -> Dict:
        """
        Calculate optimal bid/ask spread using Avellaneda-Stoikov model
        
        Args:
            mid_price: Current mid-market price
            volatility: Market volatility (annualized)
            risk_aversion: Risk aversion parameter (default: 0.1)
            time_horizon: Time horizon in years (default: 1.0)
            inventory: Current inventory position (default: 0.0)
            kappa: Order arrival rate parameter (default: 1.5)
        
        Returns:
            Dictionary with market maker quote:
            {
                "mid_price": 150.0,
                "bid": 149.25,
                "ask": 150.75,
                "spread": 1.50,
                "reservation_price": 150.0,
                "delta": 0.75,
                "model": "Avellaneda-Stoikov"
            }
        
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "mid_price": mid_price,
                "volatility": volatility,
                "risk_aversion": risk_aversion,
                "time_horizon": time_horizon,
                "inventory": inventory,
                "kappa": kappa
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/market-maker/quote",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise Exception(f"Market maker API error: {str(e)}")
    
    async def get_prediction_methods(self) -> Dict:
        """
        Get available prediction methods
        
        Returns:
            Dictionary with available methods and descriptions
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/prediction/methods",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to get prediction methods: {str(e)}")
    
    async def get_risk_levels(self) -> Dict:
        """
        Get risk level thresholds
        
        Returns:
            Dictionary with risk level definitions
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/api/risk/levels",
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Failed to get risk levels: {str(e)}")
    
    async def is_backend_available(self) -> bool:
        """
        Check if backend is available (non-throwing)
        
        Returns:
            True if backend is reachable, False otherwise
        """
        try:
            await self.health_check()
            return True
        except (httpx.HTTPError, OSError, ValueError):
            return False
    
    async def analyze_chart(self, symbol: str, data: List[Dict]) -> Dict:
        """
        Analyze stock chart data using Gemini AI
        
        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data with format:
                  [{"time": "2025-01-01", "open": 150.0, "high": 152.0, 
                    "low": 149.0, "close": 151.0, "volume": 50000}, ...]
        
        Returns:
            Dictionary with chart analysis:
            {
                "analysis": "Technical analysis text from AI"
            }
        
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "symbol": symbol,
                "data": data
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/analyze-chart",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise Exception(f"Chart analysis API error: {str(e)}")
    
    async def summarize_news(self, articles: List[str]) -> Dict:
        """
        Summarize news articles using Gemini AI
        
        Args:
            articles: List of news article texts
        
        Returns:
            Dictionary with summary:
            {
                "summary": "AI-generated summary of articles"
            }
        
        Raises:
            Exception: If API call fails
        """
        try:
            payload = {
                "articles": articles
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/summarize-news",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            raise Exception(f"News summarization API error: {str(e)}")

    async def query_gemini(self, prompt: str) -> Dict:
        """
        Query Gemini via backend helper
        
        Args:
            prompt: Prompt string
        
        Returns:
            Dictionary with response text
        """
        try:
            payload = {"prompt": prompt}
            response = await self.client.post(
                f"{self.base_url}/api/gemini-query",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise Exception(f"Gemini query API error: {str(e)}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._client:
            import warnings
            warnings.warn(
                "BackendClient was not properly closed. Use 'async with' or call close().",
                ResourceWarning
            )


# Singleton instance
_backend_client = None

def get_backend_client() -> BackendClient:
    """Get singleton Backend client instance"""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendClient()
    return _backend_client