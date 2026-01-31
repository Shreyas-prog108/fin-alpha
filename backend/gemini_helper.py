"""
Gemini API Helper - WITH GOOGLE SEARCH GROUNDING
Handles API key authentication for Google Gemini API with search capabilities
"""
import logging
from fastapi import HTTPException, status
import httpx
from backend.config import GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL, GEMINI_API_VERSION

logger = logging.getLogger(__name__)

# =========================
# GEMINI HELPER - WITH SEARCH
# =========================


async def query_gemini(prompt: str, use_search: bool = False) -> str:
    """
    Query Gemini API safely with API key authentication
    
    Args:
        prompt: The prompt to send to Gemini
        use_search: Whether to enable Google Search grounding
        
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
        # Gemini API uses API key as query parameter
        url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        headers = {
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
        
        # Add Google Search grounding if enabled
        if use_search:
            payload["tools"] = [
                {
                    "google_search": {}
                }
            ]
        
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                url_with_key,
                headers=headers,
                json=payload,
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code} - {response.text[:200]}")
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


async def query_gemini_with_search(prompt: str) -> dict:
    """
    Query Gemini API with Google Search grounding enabled.
    Returns both the response text and search results/sources.
    
    Args:
        prompt: The prompt to send to Gemini
        
    Returns:
        Dictionary with 'response' text and 'sources' list
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service temporarily unavailable"
        )
    
    try:
        url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "tools": [
                {"google_search": {}}
            ]
        }
        
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                url_with_key,
                headers=headers,
                json=payload,
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini Search API error: {response.status_code}")
                # Fallback to regular query
                text = await query_gemini(prompt, use_search=False)
                return {"response": text, "sources": [], "search_used": False}
            
            body = response.json()
            
            # Extract response text
            text = ""
            sources = []
            
            try:
                candidate = body["candidates"][0]
                text = candidate["content"]["parts"][0]["text"]
                
                # Extract grounding metadata (search sources)
                if "groundingMetadata" in candidate:
                    grounding = candidate["groundingMetadata"]
                    
                    # Get search entry point (if available)
                    if "searchEntryPoint" in grounding:
                        sources.append({
                            "type": "search",
                            "content": grounding["searchEntryPoint"].get("renderedContent", "")
                        })
                    
                    # Get grounding chunks (actual sources)
                    if "groundingChunks" in grounding:
                        for chunk in grounding["groundingChunks"]:
                            if "web" in chunk:
                                sources.append({
                                    "type": "web",
                                    "title": chunk["web"].get("title", ""),
                                    "uri": chunk["web"].get("uri", "")
                                })
                    
                    # Get web search queries used
                    if "webSearchQueries" in grounding:
                        sources.append({
                            "type": "queries",
                            "queries": grounding["webSearchQueries"]
                        })
                        
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Could not parse search response: {e}")
                text = await query_gemini(prompt, use_search=False)
            
            return {
                "response": text,
                "sources": sources,
                "search_used": True
            }
            
    except Exception as e:
        logger.error(f"Search query failed: {e}, falling back to regular query")
        text = await query_gemini(prompt, use_search=False)
        return {"response": text, "sources": [], "search_used": False}
