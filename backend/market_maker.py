from fastapi import HTTPException
import numpy as np
from models import MarketMakerRequest
from config import app


@app.post("/api/market-maker/quote")
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
