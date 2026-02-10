from fastapi import HTTPException
import numpy as np

async def predict_price(symbol: str, data: list, method: str = "ema", ema_span: int = 10) -> dict:
    if not data:
        raise HTTPException(status_code=422, detail="data must contain at least one element")

    closes_list = []
    for row in data:
        value = row.get("close", row.get("Close"))
        if value is not None:
            closes_list.append(float(value))
    if not closes_list:
        raise HTTPException(status_code=422, detail="data must include close prices")

    closes = np.array(closes_list, dtype=float)
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

    predicted_price = float(prediction)
    predicted_change = predicted_price - last_price
    predicted_change_percent = (predicted_change / last_price * 100) if last_price else 0.0

    abs_change = abs(predicted_change_percent)
    if len(closes) >= 90 and abs_change >= 2:
        confidence = "high"
    elif len(closes) >= 30:
        confidence = "medium"
    else:
        confidence = "low"

    trend = "bullish" if predicted_change > 0 else ("bearish" if predicted_change < 0 else "neutral")

    return {
        "symbol": symbol,
        "method": method,
        "current_price": last_price,
        "predicted_price": predicted_price,
        "predicted_change": float(predicted_change),
        "predicted_change_percent": float(predicted_change_percent),
        "confidence": confidence,
        "trend": trend,
        "last_price": last_price,
        "prediction": predicted_price,
        "details": details,
    }
