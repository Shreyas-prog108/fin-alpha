"""
Gemini API Helper - SECURITY HARDENED
Handles API key securely via headers instead of URL parameters
"""
import logging
from fastapi import HTTPException, status
import httpx
from backend.config import GEMINI_API_KEY, GEMINI_API_URL

logger = logging.getLogger(__name__)

# =========================
# GEMINI HELPER - SECURE
# =========================


async def query_gemini(prompt: str) -> str:
    """
    Query Gemini API safely with secure header-based authentication
    
    Args:
        prompt: The prompt to send to Gemini
        
    Returns:
        The response text from Gemini
        
    Raises:
        HTTPException: If API call fails
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service temporarily unavailable"
        )
    
    if not prompt or len(prompt) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty"
        )
    
    if len(prompt) > 100000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt too long (max 100000 characters)"
        )
    
    try:
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Service temporarily unavailable"
                )
            
            body = response.json()
            
            try:
                return body["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Unexpected Gemini response structure: {type(e).__name__}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Service temporarily unavailable"
                )
                
    except httpx.TimeoutException:
        logger.error("Gemini API timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Service temporarily unavailable"
        )
    except httpx.HTTPError as e:
        logger.error(f"Gemini API HTTP error: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )
    except Exception as e:
        logger.exception(f"Unexpected error in query_gemini: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
