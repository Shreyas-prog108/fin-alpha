from typing import Optional
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import httpx
import os

#=========================
#REQUEST MODELS - WITH VALIDATION
#=========================
SYMBOL_REGEX = r"^[A-Z0-9&-]{1,15}(\.[A-Z]{1,4})?$"


class NewsRequest(BaseModel):
    articles: list[str] = Field(..., min_items=1, max_items=100)
    
    @field_validator('articles')
    @classmethod
    def validate_articles(cls, v):
        for article in v:
            if not isinstance(article, str) or len(article) == 0 or len(article) > 10000:
                raise ValueError('Each article must be a non-empty string (max 10000 chars)')
        return v


class ChartRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    data: list[dict] = Field(..., min_items=1, max_items=1000)
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.strip().upper()
        if not re.match(SYMBOL_REGEX, v):
            raise ValueError('Invalid stock symbol format')
        return v


class RiskAnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    data: list[dict] = Field(..., min_items=1, max_items=1000)
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.strip().upper()
        if not re.match(SYMBOL_REGEX, v):
            raise ValueError("Invalid stock symbol format")
        return v


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    data: list[dict] = Field(..., min_items=1, max_items=1000)
    method: Optional[str] = Field("ema", pattern="^(ema|linear)$")
    horizon: Optional[int] = Field(1, ge=1, le=365)
    ema_span: Optional[int] = Field(10, ge=2, le=100)
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.strip().upper()
        if not re.match(SYMBOL_REGEX, v):
            raise ValueError("Invalid stock symbol format")
        return v


class MarketMakerRequest(BaseModel):
    mid_price: float = Field(..., gt=0, le=1000000)
    volatility: float = Field(..., gt=0, lt=10)
    risk_aversion: float = Field(0.1, ge=0.001, le=10)
    time_horizon: float = Field(1.0, gt=0, le=365)
    inventory: float = Field(0.0, ge=-1000000, le=1000000)
    kappa: float = Field(1.5, gt=0, le=100)
    max_spread: Optional[float] = Field(None, gt=0, le=1000000)


class GroqQueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)


class SearchAnalysisRequest(BaseModel):
    """Request model for stock analysis"""
    symbol: str = Field(..., min_length=1, max_length=20)
    company_name: str = Field(..., min_length=1, max_length=100)
    query_type: str = Field(default="analysis", description="Type: analysis, news, sentiment, recommendation")
    time_frame: str = Field(default="3mo", description="Time frame for analysis")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.strip().upper()
        if not re.match(SYMBOL_REGEX, v):
            raise ValueError("Invalid stock symbol format")
        return v


class NewsAnalysisRequest(BaseModel):
    """Request model for combined news analysis"""
    symbol: str = Field(..., min_length=1, max_length=20)
    company_name: str = Field(..., min_length=1, max_length=100)
    newsapi_articles: list[dict] = Field(default_factory=list, max_length=10)
    mint_articles: list[dict] = Field(default_factory=list, max_length=10)
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        v = v.strip().upper()
        if not re.match(SYMBOL_REGEX, v):
            raise ValueError('Invalid stock symbol format')
        return v
