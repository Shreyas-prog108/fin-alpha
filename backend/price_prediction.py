from fastapi import HTTPException
import numpy as np

async def predict_price(symbol: str, data: list, method: str = "ema", ema_span: int = 10) -> dict:
    if not data:
        raise HTTPException(status_code=422, detail="data must contain at least one element")
    closes = np.array([d["close"] for d in data])
    last_price = float(closes[-1])
    method = method.lower()
    prediction = last_price
    details: dict = {}
    if method == "ema":
        span = int(ema_span or 10)
        alpha = 2.0 / (span + 1.0)
        ema = float(closes[0])
        for p in closes[1:]:
            ema = alpha * float(p) + (1 - alpha) * ema
        prediction = ema
        details = {"ema": float(ema), 'alpha': float(alpha)}
    elif method == "linear":
        n = len(closes)
        t = np.arange(n, dtype=float)
        A = np.vstack([t, np.ones(n)]).T
        a, b = np.linalg.lstsq(A, closes.astype(float), rcond=None)[0]
        prediction = float(a * n + b)
        residuals = closes - (a * t + b)
        sigma = float(np.sqrt(np.mean(residuals ** 2)))
        details = {"slope": float(a), "intercept": float(b), "residual_sigma": sigma}
    else:
        raise HTTPException(status_code=422, detail="method must be 'ema' or 'linear'")
    return {"symbol": symbol, "method": method, "last_price": last_price, "prediction": float(prediction), "details": details}