from fastapi import HTTPException
from models import ChartRequest
from config import app
from gemini_helper import query_gemini

@app.post("/api/analyze-chart")
async def analyze_chart(req:ChartRequest)->dict:
    prompt = f"""
    Analyze the stock chart data for {req.symbol}.
    Data: {req.data}

    Provide:
    - Trend direction (bullish, bearish, neutral)
    - Strong signals (volume spikes, breakouts, moving averages implied)
    - Risks or anomalies
    - A one-sentence summary for traders
    """
    analysis=await query_gemini(prompt)
    return {"analysis":analysis}