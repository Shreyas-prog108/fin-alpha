from fastapi import HTTPException
import numpy as np
from models import PredictionRequest
from config import app


@app.post("/api/predict-price",tags=["Prediction"])
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