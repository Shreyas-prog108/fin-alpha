"""
fin-alpha FastAPI Application - SECURITY HARDENED
Provides market analysis, risk scoring, price prediction, and market maker quoting
"""
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_API_VERSION,
    ALLOWED_ORIGINS,
    GEMINI_API_KEY,
    REQUIRE_HTTPS,
)
from backend.models import (
    NewsRequest,
    ChartRequest,
    PredictionRequest,
    RiskAnalysisRequest,
    MarketMakerRequest,
)
from backend.risk_analysis import analyze_risk
from backend.market_maker import market_maker_quote
from backend.price_prediction import predict_price

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with security configs
app = FastAPI(
    title="fin-alpha - Financial Market Analysis API",
    description="Secure Market Analysis & Prediction Service",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================
# SECURITY MIDDLEWARE
# =============================

# 1. Add CORS middleware with strict configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# 2. Add trusted host middleware (prevents Host header attacks)
allowed_hosts = [origin.split("://")[-1] for origin in ALLOWED_ORIGINS if origin.strip()]
allowed_hosts.extend(["localhost", "127.0.0.1", "localhost:8000", "127.0.0.1:8000"])
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)

# 3. Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Prevent content type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # HSTS (only in production with HTTPS)
    if REQUIRE_HTTPS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # CSP (Content Security Policy)
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

# =============================
# AUTHENTICATION
# =============================

security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify API key from Authorization header
    Format: Authorization: Bearer <api_key>
    Optional for development mode
    """
    if not credentials:
        logger.info("Request without auth - allowing in dev mode")
        return "dev-mode"
    
    return credentials.credentials


# =============================
# PUBLIC ENDPOINTS (No auth required)
# =============================

@app.get("/api/health")
async def health_check():
    """Health check endpoint - no authentication required"""
    return {
        "status": "ok",
        "service": "fin-alpha API",
        "version": "2.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint - no authentication required"""
    return {
        "status": "ok",
        "service": "fin-alpha API",
        "docs": "/docs" ,
        "version": "2.0"
    }


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


# =============================
# PROTECTED ENDPOINTS (Auth required)
# =============================

@app.get("/api/gemini-models")
async def list_gemini_models(api_key: str = Depends(verify_api_key)):
    """
    List available Gemini models
    Requires: Authorization: Bearer <API_KEY>
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service temporarily unavailable"
        )
    
    try:
        url = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models"
        
        # SECURITY FIX: Use Authorization header instead of URL parameters
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, headers=headers)
            
            if resp.status_code != 200:
                logger.error(f"Gemini API error: {resp.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Service temporarily unavailable"
                )
            
            return resp.json()
            
    except httpx.TimeoutException:
        logger.error("Gemini API timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Service temporarily unavailable"
        )
    except Exception as e:
        logger.exception(f"Unexpected error in list_gemini_models: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/analyze-risk")
async def risk_analysis(
    request: RiskAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze stock risk and detect anomalies
    Requires: Authorization: Bearer <API_KEY>
    """
    try:
        logger.info(f"Risk analysis requested for {request.symbol}")
        result = await analyze_risk(request.symbol, request.data)
        return result
    except ValueError as e:
        logger.warning(f"Validation error in risk_analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error in risk_analysis: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/predict-price")
async def price_prediction(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict stock price using EMA or linear regression
    Requires: Authorization: Bearer <API_KEY>
    """
    try:
        logger.info(f"Price prediction requested for {request.symbol} using {request.method}")
        result = await predict_price(
            request.symbol,
            request.data,
            request.method,
            request.ema_span
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in price_prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error in price_prediction: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/market-maker/quote")
async def market_maker_endpoint(
    request: MarketMakerRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get optimal bid/ask spread using Avellaneda-Stoikov model
    Requires: Authorization: Bearer <API_KEY>
    """
    try:
        logger.info(f"Market maker quote requested for mid_price={request.mid_price}")
        result = market_maker_quote(
            request.mid_price,
            request.volatility,
            request.risk_aversion,
            request.time_horizon,
            request.inventory,
            request.kappa,
            request.max_spread
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in market_maker: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error in market_maker: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# =============================
# ERROR HANDLERS
# =============================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors gracefully"""
    logger.warning(f"Validation error: {str(exc)}")
    return {
        "error": "Invalid request",
        "status_code": 400
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors gracefully"""
    logger.exception(f"Unexpected error: {type(exc).__name__}")
    return {
        "error": "Internal server error",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
