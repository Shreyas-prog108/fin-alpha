import numpy as np

async def analyze_risk(symbol: str, data: list) -> dict:
    closes = np.array([float(d.get("close", 0)) for d in data], dtype=float)
    volumes = np.array([float(d.get("volume", 0)) for d in data], dtype=float)
    if closes.size == 0:
        return {"symbol": symbol, "error": "No close prices provided"}

    returns = np.diff(closes) / closes[:-1] if closes.size > 1 else np.array([])
    volatility = float(np.std(returns) * np.sqrt(252)) if returns.size > 1 else 0.0

    vol_mean, vol_std = np.mean(volumes), np.std(volumes)
    anomalies = []
    if vol_std > 0:
        for i, v in enumerate(volumes):
            z = (v - vol_mean) / vol_std
            if abs(z) > 2:
                timestamp = data[i].get("time") or data[i].get("date") or str(i)
                anomalies.append({"time": timestamp, "volume": float(v), "z_score": float(z)})

    risk_score = "LOW"
    if volatility > 0.05:
        risk_score = "MEDIUM"
    if volatility > 0.15:
        risk_score = "HIGH"

    max_drawdown = 0.0
    if closes.size > 1:
        running_max = np.maximum.accumulate(closes)
        drawdowns = (closes - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

    sharpe_ratio = 0.0
    if returns.size > 1 and np.std(returns) > 0:
        sharpe_ratio = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))

    var_95 = float(np.percentile(returns, 5)) if returns.size > 1 else 0.0
    risk_level = risk_score.lower()

    return {
        "symbol": symbol,
        "volatility": volatility,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "var_95": var_95,
        "beta": None,
        "anomalies": anomalies,
    }
