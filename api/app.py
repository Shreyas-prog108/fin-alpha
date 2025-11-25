from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import numpy as np
from dotenv import load_dotenv
from typing import Union, Optional
from api.config import GEMINI_API_KEY, GEMINI_API_VERSION
from api.models import NewsRequest, ChartRequest, PredictionRequest, MarketMakerRequest
from api.risk_analysis import analyze_risk
from api.market_maker import market_maker_quote
from api.price_prediction import predict_price

load_dotenv()

app=FastAPI(title="FinTerm - The Next Generation of Stock Market Analysis",description="Market Analysis & Prediction Agent",version="1.1")


#=========================
#ENTRYPOINTS
#=========================
@app.get("/api/health")
async def health():
    return {"status":"ok","service":"Agent Server"}


@app.get("/")
async def root():
    return {"status":"ok","service":"Agent Server","docs": "/docs"}


@app.get("/api/gemini-models")
async def list_gemini_models():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500,detail="GEMINI_API_KEY is not set")
    url=f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models"
    async with httpx.AsyncClient(timeout=60) as client:
        resp=await client.get(url, params={"key":GEMINI_API_KEY})
        if resp.status_code!=200:
            raise HTTPException(status_code=500,detail=f"Gemini API error:{resp.text}")
        return resp.json()

