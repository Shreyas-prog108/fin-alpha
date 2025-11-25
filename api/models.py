from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

#=========================
#REQUEST MODELS
#=========================

class NewsRequest(BaseModel):
    articles:list[str]

class ChartRequest(BaseModel):
    symbol:str
    data:list[dict]


class PredictionRequest(BaseModel):
    symbol:str
    data:list[dict]
    method:Optional[str]="ema"
    horizon:Optional[int]=1
    ema_span:Optional[int]=10

class MarketMakerRequest(BaseModel):
    mid_price:float
    volatility:float
    risk_aversion:float
    time_horizon:float
    inventory:float
    kappa:float
    max_spread:Optional[float]=None