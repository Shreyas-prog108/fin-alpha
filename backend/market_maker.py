from fastapi import HTTPException
import numpy as np

def market_maker_quote(mid_price: float, volatility: float, risk_aversion: float = 0.1, 
                       time_horizon: float = 1.0, inventory: float = 0.0, kappa: float = 1.5,
                       max_spread: float = 0.0) -> dict:
    if risk_aversion <= 0 or kappa <= 0 or time_horizon <= 0 or volatility < 0:
        raise HTTPException(status_code=422, detail="risk_aversion, kappa, time_horizon must be >0 and volatility >=0")
    gamma = float(risk_aversion)
    sigma = float(volatility)
    T = float(time_horizon)
    q = float(inventory)
    m = float(mid_price)
    kappa_val = float(kappa)
    reservation_price = m - q * gamma * sigma ** 2 * T
    spread = gamma * sigma ** 2 * T + (2.0 / gamma) * np.log(1.0 + (gamma / kappa_val))
    if max_spread > 0:
        spread = float(min(spread, max_spread))
    bid = reservation_price - spread / 2.0
    ask = reservation_price + spread / 2.0
    
    # Additional calculated fields
    spread_percent = (spread / m * 100) if m > 0 else 0
    bid_price = float(bid)
    ask_price = float(ask)
    
    # Inventory adjustment direction
    inventory_adjustment = "long_bias" if q > 0 else ("short_bias" if q < 0 else "neutral")
    
    # Quote recommendation
    if abs(q) < m * 0.01:  # Less than 1% inventory
        quote_recommendation = "Standard quotes - balanced inventory"
    elif q > 0:
        quote_recommendation = "Tighten ask / widen bid - reduce long exposure"
    else:
        quote_recommendation = "Tighten bid / widen ask - reduce short exposure"
    
    return {
        "mid_price": m,
        "reservation_price": float(reservation_price),
        "optimal_spread": float(spread),
        "spread": float(spread),
        "spread_percent": round(spread_percent, 4),
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid": float(bid),
        "ask": float(ask),
        "volatility_used": sigma,
        "risk_aversion": gamma,
        "kappa": kappa_val,
        "inventory_adjustment": inventory_adjustment,
        "quote_recommendation": quote_recommendation,
        "params": {"gamma": gamma, "sigma": sigma, "T": T, "inventory": q, "kappa": kappa_val}
    }
