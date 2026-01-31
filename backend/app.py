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
    GeminiQueryRequest,
    GeminiSearchAnalysisRequest,
    NewsAnalysisRequest,
)
from backend.risk_analysis import analyze_risk
from backend.market_maker import market_maker_quote
from backend.price_prediction import predict_price
from backend.gemini_helper import query_gemini

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# 1.CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# 2.middleware
allowed_hosts = [origin.split("://")[-1] for origin in ALLOWED_ORIGINS if origin.strip()]
allowed_hosts.extend(["localhost", "127.0.0.1", "localhost:8000", "127.0.0.1:8000"])
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)

# 3.security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    if REQUIRE_HTTPS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
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


@app.post("/api/gemini-query")
async def gemini_query_endpoint(
    request: GeminiQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Proxy a prompt to Gemini via backend helper
    Optionally enable Google Search grounding with use_search=true
    """
    try:
        if request.use_search:
            from backend.gemini_helper import query_gemini_with_search
            result = await query_gemini_with_search(request.prompt)
            return result
        else:
            response_text = await query_gemini(request.prompt)
            return {"response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in gemini_query: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.post("/api/search-analysis")
async def search_grounded_analysis(
    request: GeminiSearchAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Perform comprehensive stock analysis using Gemini with Google Search grounding.
    This retrieves real-time news and market data via search, then provides AI analysis.
    """
    try:
        from backend.gemini_helper import query_gemini_with_search
        
        logger.info(f"Search analysis requested for {request.symbol}")
        
        # Build comprehensive search prompt
        prompt = f"""You are a financial analyst. Use Google Search to find the latest information and provide a comprehensive analysis.

**Stock:** {request.company_name} ({request.symbol})
**Analysis Type:** {request.query_type}
**Time Frame:** {request.time_frame}

Please search for and analyze:

1. **Latest News** (last 7 days):
   - Search for recent news about {request.company_name}
   - Include any earnings reports, management changes, or major announcements
   
2. **Stock Price & Performance**:
   - Current stock price and recent price movement
   - Compare with sector performance
   
3. **Market Sentiment**:
   - Analyst ratings and price targets
   - Social media and investor sentiment
   
4. **Key Events & Catalysts**:
   - Upcoming earnings dates
   - Any regulatory or legal developments
   - Product launches or business developments

5. **Investment Recommendation**:
   Based on your search findings, provide:
   - BUY / HOLD / SELL recommendation
   - Key reasons for your recommendation
   - Risk factors to consider
   - Price target (if available from analyst consensus)

Format your response with clear sections and bullet points. Include specific dates and numbers where available. Cite your sources."""

        # Query Gemini with search grounding
        result = await query_gemini_with_search(prompt)
        
        return {
            "symbol": request.symbol,
            "company_name": request.company_name,
            "query_type": request.query_type,
            "time_frame": request.time_frame,
            "analysis": result.get("response", ""),
            "sources": result.get("sources", []),
            "search_used": result.get("search_used", False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in search analysis: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search analysis failed"
        )


@app.post("/api/analyze-news")
async def analyze_news(
    request: NewsAnalysisRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Analyze combined news from NewsAPI and LiveMint using Gemini
    Returns sentiment analysis and investment insights
    """
    try:
        logger.info(f"News analysis requested for {request.symbol}")
        
        # Prepare articles summary
        all_articles = []
        
        # Process NewsAPI articles
        for article in request.newsapi_articles[:3]:
            all_articles.append({
                "source": article.get("source", "NewsAPI"),
                "title": article.get("title", ""),
                "summary": article.get("description", article.get("summary", ""))[:500],
                "sentiment": article.get("sentiment", "neutral"),
                "sentiment_score": article.get("sentiment_score", 0)
            })
        
        # Process LiveMint articles
        for article in request.mint_articles[:3]:
            all_articles.append({
                "source": "LiveMint",
                "title": article.get("title", ""),
                "summary": article.get("summary", "")[:500],
                "sentiment": article.get("sentiment", "neutral"),
                "sentiment_score": article.get("sentiment_score", 0)
            })
        
        if not all_articles:
            return {
                "symbol": request.symbol,
                "company_name": request.company_name,
                "analysis": "No recent news articles found for analysis.",
                "sentiment_summary": "neutral",
                "confidence": 0.0,
                "articles_analyzed": 0
            }
        
        # Build prompt for Gemini
        articles_text = "\n".join([
            f"- [{a['source']}] {a['title']}: {a['summary'][:200]}... (Sentiment: {a['sentiment']})"
            for a in all_articles
        ])
        
        prompt = f"""Analyze the following recent news articles about {request.company_name} ({request.symbol}) and provide:

1. **Overall Sentiment**: Is the news generally positive, negative, or mixed for the stock?
2. **Key Themes**: What are the main topics/events being discussed?
3. **Investment Implications**: How might this news affect the stock price in the short term?
4. **Risk Factors**: Any concerns or risks highlighted in the news?
5. **Confidence Level**: How confident are you in this analysis (low/medium/high)?

News Articles:
{articles_text}

Provide a concise analysis (max 300 words) that would help an investor make informed decisions."""
        
        # Get Gemini analysis
        analysis = await query_gemini(prompt)
        
        # Calculate aggregate sentiment
        sentiments = [a["sentiment"] for a in all_articles]
        scores = [a["sentiment_score"] for a in all_articles if a["sentiment_score"] != 0]
        
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        
        if positive > negative:
            overall_sentiment = "positive"
        elif negative > positive:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "symbol": request.symbol,
            "company_name": request.company_name,
            "analysis": analysis,
            "sentiment_summary": overall_sentiment,
            "sentiment_score": round(avg_score, 2),
            "sentiment_breakdown": {
                "positive": positive,
                "negative": negative,
                "neutral": len(sentiments) - positive - negative
            },
            "articles_analyzed": len(all_articles),
            "sources": {
                "newsapi": len(request.newsapi_articles[:3]),
                "livemint": len(request.mint_articles[:3])
            },
            "top_headlines": [a["title"] for a in all_articles[:3]]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in news analysis: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze news"
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
