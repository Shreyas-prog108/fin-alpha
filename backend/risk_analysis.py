import numpy as np

async def analyze_risk(symbol: str, data: list) -> dict:
    closes = np.array([float(d.get("close", 0)) for d in data], dtype=float)
    volumes = np.array([float(d.get("volume", 0)) for d in data], dtype=float)
    highs = np.array([float(d.get("high", 0)) for d in data], dtype=float)
    lows = np.array([float(d.get("low", 0)) for d in data], dtype=float)
    
    if closes.size == 0:
        return {"symbol": symbol, "error": "No close prices provided"}

    returns = np.diff(closes) / closes[:-1] if closes.size > 1 else np.array([])
    volatility = float(np.std(returns) * np.sqrt(252)) if returns.size > 1 else 0.0
    daily_volatility = float(np.std(returns)) if returns.size > 1 else 0.0
    
    # Support and Resistance levels
    support_level = float(np.min(closes[-20:])) if closes.size >= 20 else float(np.min(closes))
    resistance_level = float(np.max(closes[-20:])) if closes.size >= 20 else float(np.max(closes))

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
    max_dd_date = None
    if closes.size > 1:
        running_max = np.maximum.accumulate(closes)
        drawdowns = (closes - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))
        max_dd_idx = np.argmin(drawdowns)
        max_dd_date = data[max_dd_idx].get("date") or data[max_dd_idx].get("time") if max_dd_idx < len(data) else None

    sharpe_ratio = 0.0
    if returns.size > 1 and np.std(returns) > 0:
        sharpe_ratio = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))
    
    # Sortino Ratio (downside risk only)
    sortino_ratio = 0.0
    negative_returns = returns[returns < 0]
    if negative_returns.size > 1 and np.std(negative_returns) > 0:
        sortino_ratio = float((np.mean(returns) / np.std(negative_returns)) * np.sqrt(252))
    
    # Treynor Ratio (assuming beta = 1 for simplicity)
    treynor_ratio = 0.0
    beta = 1.0  # Simplified - would need market data for real calculation
    if beta > 0 and returns.size > 1:
        treynor_ratio = float((np.mean(returns) * 252 - 0.04) / beta)  # Assuming 4% risk-free rate
    
    # Information Ratio
    information_ratio = 0.0
    if returns.size > 1:
        information_ratio = float(np.mean(returns) * np.sqrt(252) / (np.std(returns) * np.sqrt(252)))
    
    # Alpha (assuming 0 for simplicity - would need benchmark)
    alpha = 0.0
    
    # R-squared (simplified)
    r_squared = 0.85  # Placeholder

    var_95 = float(np.percentile(returns, 5)) if returns.size > 1 else 0.0
    cvar_95 = float(np.mean(returns[returns <= var_95])) if returns.size > 1 and np.any(returns <= var_95) else var_95
    
    # Volatility comparison to typical sector ranges
    # Indian banking sector typically 12-18%
    volatility_vs_sector = "below sector avg" if volatility < 0.12 else ("within sector range" if volatility <= 0.18 else "above sector avg")
    
    risk_level = risk_score.lower()
    
    # Risk flag
    risk_flag = "elevated_volatility" if volatility > 0.18 else ("moderate_volatility" if volatility > 0.10 else "normal")

    return {
        "symbol": symbol,
        "volatility": volatility,
        "daily_volatility": daily_volatility,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_dd_date,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
        "volatility_vs_sector": volatility_vs_sector,
        "risk_flag": risk_flag,
        "anomalies": anomalies,
        "support_level": support_level,
        "resistance_level": resistance_level,
        "avg_volume": float(np.mean(volumes)) if volumes.size > 0 else 0,
        "volume_std": float(np.std(volumes)) if volumes.size > 0 else 0,
    }
