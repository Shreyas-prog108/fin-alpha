from fastapi import HTTPException
from models import NewsRequest
from config import app
from gemini_helper import query_gemini

@app.post("/api/summarize-news")
async def summarize_news(req:NewsRequest)->dict:
    joined="\n\n".join(req.articles)
    prompt=f"""
    Summarize the following financial news articles into a concise market summary.
    Focus on sentiment (bullish, bearish, neutral), key events, and risks:
    """
    summary=await query_gemini(prompt)
    return {"summary":summary}