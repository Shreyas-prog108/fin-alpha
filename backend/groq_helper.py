"""
Groq API Helper
Handles API key authentication for Groq API
"""
import logging
from fastapi import HTTPException, status
from groq import AsyncGroq
from backend.config import GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL

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
    Query Gemini with Google Search grounding for real-time grounded answers.
    Falls back to plain Groq if GOOGLE_API_KEY is unavailable.
    Returns response text, sources, and search_used flag.
    """
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                tools=[{"google_search": {}}],
            )
            response = model.generate_content(prompt)
            text = response.text or ""

            # Extract grounding sources
            sources: list[str] = []
            try:
                for candidate in response.candidates:
                    gm = getattr(candidate, "grounding_metadata", None)
                    if gm is None:
                        continue
                    for chunk in (getattr(gm, "grounding_chunks", None) or []):
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", "") or ""
                            if uri:
                                sources.append(uri)
            except Exception:
                pass

            return {
                "response": text,
                "sources": sources,
                "search_used": True,
            }
        except Exception as e:
            logger.warning(f"Gemini search failed, falling back to Groq: {e}")

    # Fallback: plain Groq (no search)
    text = await query_groq(prompt)
    return {
        "response": text,
        "sources": [],
        "search_used": False,
    }
