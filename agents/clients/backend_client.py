"""
Backend API Client
Client for calling FinAgent FastAPI backend
"""

import httpx
import os
from typing import Dict, List, Optional


class BackendClient:
    """
    Client for FinAgent FastAPI backend
    Calls quantitative analysis endpoints
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.timeout = 30.0
        self._client = None
    
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
    
    async def health_check(self) -> Dict:
        """
        Check if backend is running
        
        Returns:
            Dictionary with service status
        
        Raises:
            Exception: If backend is unreachable
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/health")
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
                f"{self.base_url}/api/v1/risk/analyze",
                json=payload
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
                f"{self.base_url}/api/v1/prediction/predict",
                json=payload
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
                f"{self.base_url}/api/v1/market-maker/quote",
                json=payload
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
                f"{self.base_url}/api/v1/prediction/methods"
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
                f"{self.base_url}/api/v1/risk/levels"
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
        except:
            return False
    
    def __del__(self):
        """Cleanup on deletion"""
        if self._client:
            try:
                import asyncio
                asyncio.create_task(self._client.aclose())
            except:
                pass


# Singleton instance
_backend_client = None

def get_backend_client() -> BackendClient:
    """Get singleton Backend client instance"""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendClient()
    return _backend_client