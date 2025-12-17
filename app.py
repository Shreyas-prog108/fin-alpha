from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import numpy as np
from dotenv import load_dotenv
from typing import Union, Optional

load_dotenv()

#=========================
#CONFIG
#=========================
GEMINI_API_VERSION=os.getenv("GEMINI_API_VERSION","v1beta")
GEMINI_MODEL=os.getenv("GEMINI_MODEL","gemini-3-flash-preview")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
GEMINI_API_URL=(
    f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/"
    f"{GEMINI_MODEL}:generateContent"
)
app=FastAPI(title="FinTerm - Terminal-Based MCP Server for Stock Market Analysis",description="Market Analysis & Prediction Engine",version="1.1")

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

#=========================
#GEMINI HELPER
#=========================
async def query_gemini(prompt:str)->str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500,detail="GEMINI_API_KEY is not set")
    async with httpx.AsyncClient(timeout=60) as client:
        response=await client.post(
            GEMINI_API_URL,
            params={"key":GEMINI_API_KEY},
            json={"contents":[{"parts":[{"text":prompt}]}]},
        )
        if response.status_code!=200:
            raise HTTPException(status_code=500,detail=f"Gemini API error: {response.text}")
        body=response.json()
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise HTTPException(status_code=500,detail=f"Unexpected Gemini response:{body}")      


#=========================
#ENTRYPOINTS
#=========================
@app.get("/api/v1/mcp/health")
async def health():
    return {"status":"ok","service":"MCP Server"}


@app.get("/")
async def root():
    return {"status":"ok","service":"MCP Server","docs": "/docs"}

#=========================
#ENDPOINTS
#=========================
@app.post("/api/v1/mcp/summarize-news")
async def summarize_news(req:NewsRequest)->dict:
    joined="\n\n".join(req.articles)
    prompt=f"""
    Summarize the following financial news articles into a concise market summary.
    Focus on sentiment (bullish, bearish, neutral), key events, and risks:
    """
    summary=await query_gemini(prompt)
    return {"summary":summary}

@app.post("/api/v1/mcp/analyze-chart")
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

@app.get("/api/v1/mcp/models")
async def list_gemini_models():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500,detail="GEMINI_API_KEY is not set")
    url=f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models"
    async with httpx.AsyncClient(timeout=60) as client:
        resp=await client.get(url, params={"key":GEMINI_API_KEY})
        if resp.status_code!=200:
            raise HTTPException(status_code=500,detail=f"Gemini API error:{resp.text}")
        return resp.json()

@app.post("/api/v1/mcp/analyze-risk")
async def analyze_risk(req:ChartRequest)->dict:
    closes=np.array([d["close"] for d in req.data])
    volumes=np.array([d["volume"] for d in req.data])
    volatility=float(np.std(closes)/np.mean(closes)) if np.mean(closes)!=0 else 0.0
    vol_mean,vol_std=np.mean(volumes),np.std(volumes)
    anomalies=[]
    if vol_std>0:
        for i,v in enumerate(volumes):
            z=(v-vol_mean)/vol_std
            if abs(z)>2:
                anomalies.append({"time": req.data[i]["time"],"volume": float(v),"z_score": float(z)})
    risk_score='LOW'
    if volatility>0.05:
        risk_score='MEDIUM'
    if volatility>0.15:
        risk_score='HIGH'
    return {"symbol": req.symbol,"volatility": volatility,"risk_score": risk_score,"anomalies": anomalies}


@app.post("/api/v1/mcp/predict-price")
async def predict_price(req:PredictionRequest)->dict:
    if not req.data:
        raise HTTPException(status_code=422,detail="data must contain at least one element")
    closes=np.array([d["close"] for d in req.data])
    last_price=float(closes[-1])
    method=(req.method or "ema").lower()
    prediction=last_price
    details:dict[str,float]={}
    if method=="ema":
        span=int(req.ema_span or 10)
        alpha=2.0/(span+1.0)
        ema=float(closes[0])
        for p in closes[1:]:
            ema=alpha*float(p)+(1-alpha)*ema
        prediction=ema
        details={"ema":float(ema),'alpha':float(alpha)}
    elif method=="linreg":
        n=len(closes)
        t=np.arange(n,dtype=float)
        A=np.vstack([t,np.ones(n)]).T
        a,b=np.linalg.lstsq(A,closes.astype(float),rcond=None)[0]
        prediction=float(a*n+b)
        residuals=closes-(a*t+b)
        sigma=float(np.sqrt(np.mean(residuals**2)))
        details={"slope":float(a),"intercept":float(b),"residual_sigma":sigma}
    else:
        raise HTTPException(status_code=422,detail="method must be 'ema' or 'linreg'")
    return{ "symbol": req.symbol,"method": method,"last_price": last_price,"prediction": float(prediction),"horizon": int(req.horizon or 1),"details": details}

@app.post("/api/v1/mcp/market-maker/quote")
async def market_maker_quote(req:MarketMakerRequest)->dict:
    if req.risk_aversion<=0 or req.kappa<=0 or req.time_horizon<=0 or req.volatility<0:
        raise HTTPException(status_code=422,detail="risk_aversion,kappa,T must be >0 and volatility >=0")
    gamma=float(req.risk_aversion)
    sigma=float(req.volatility)
    T=float(req.time_horizon)
    q=float(req.inventory)
    m=float(req.mid_price)
    kappa=float(req.kappa)
    reservation_price=m-q*gamma*sigma**2*T
    spread=gamma*sigma**2*T+(2.0/gamma)*np.log(1.0+(gamma/kappa))
    if req.max_spread is not None:
        spread=float(min(spread,req.max_spread))
    bid=reservation_price-spread/2.0
    ask=reservation_price+spread/2.0
    return {"mid_price": m,"reservation_price": float(reservation_price),"optimal_spread": float(spread),"bid": float(bid),"ask": float(ask),"params": {"gamma": gamma,"sigma": sigma,"T": T,"inventory": q,"kappa": kappa}}
