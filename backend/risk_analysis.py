from fastapi import HTTPException
import numpy as np
from models import ChartRequest
from config import app

@app.post("/api/analyze-risk")
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
