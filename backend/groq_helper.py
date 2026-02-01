"""
Groq API Helper
Handles API key authentication for Groq API
"""
import logging
from fastapi import HTTPException, status
import httpx
from groq import AsyncGroq
from backend.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# =========================
# GROQ HELPER
# =========================

async def query_groq(prompt: str, use_search: bool = False) -> str:
    """
    Query Groq API safely with API key authentication
    
    Args:
        prompt: The prompt to send to Groq
        use_search: Included for compatibility, but Groq doesn't support native search grounding currently.
        
    Returns:
        The response text from Groq
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service temporarily unavailable (Groq Key missing)"
        )
    
    if not prompt or len(prompt) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty"
        )
    
    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        
        # Note: Groq doesn't support search grounding natively like Gemini does.
        # ignoring use_search for now.
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL,
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        logger.exception(f"Groq API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API Error: {str(e)}"
        )


async def query_groq_with_search(prompt: str) -> dict:
    """
    Query Groq API. Search grounding is currently NOT supported on Groq.
    Returns response text and empty sources for compatibility.
    """
    text = await query_groq(prompt)
    return {
        "response": text,
        "sources": [],
        "search_used": False
    }
