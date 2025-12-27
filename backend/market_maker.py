from fastapi import HTTPException
import numpy as np

def market_maker_quote(mid_price: float, volatility: float, risk_aversion: float = 0.1, 
                       time_horizon: float = 1.0, inventory: float = 0.0, kappa: float = 1.5,
                       max_spread: float = None) -> dict:
    if risk_aversion <= 0 or kappa <= 0 or time_horizon <= 0 or volatility < 0:
        raise HTTPException(status_code=422, detail="risk_aversion, kappa, time_horizon must be >0 and volatility >=0")
    gamma = float(risk_aversion)
    sigma = float(volatility)
    T = float(time_horizon)
    q = float(inventory)
    m = float(mid_price)
    kappa = float(kappa)
    reservation_price = m - q * gamma * sigma ** 2 * T
    spread = gamma * sigma ** 2 * T + (2.0 / gamma) * np.log(1.0 + (gamma / kappa))
    if max_spread is not None:
        spread = float(min(spread, max_spread))
    bid = reservation_price - spread / 2.0
    ask = reservation_price + spread / 2.0
    return {"mid_price": m, "reservation_price": float(reservation_price), "optimal_spread": float(spread), 
            "bid": float(bid), "ask": float(ask), 
            "params": {"gamma": gamma, "sigma": sigma, "T": T, "inventory": q, "kappa": kappa}}
