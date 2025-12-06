from __future__ import annotations
from typing import Dict, Any, Literal
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult, _safe_last, _pick_style,
    compute_basic_ohlc_aliases, compute_atr, compute_rsi,
    Direction
)

class OptionsQuantEngine:
    MODEL_VERSION = "options_directional_v2.0"

    @classmethod
    def run(
        cls, 
        df: pd.DataFrame, 
        symbol: str, 
        trade_style: str | None = None,
        # Optional params for future expansion
        side: Literal["CALL", "PUT"] = "CALL" 
    ) -> SignalResult:
        
        if df is None or df.empty: raise ValueError("Empty Data")
        
        style = _pick_style(trade_style)
        df = compute_basic_ohlc_aliases(df)
        close = _safe_last(df["Close"])
        
        # 1. Directional Edge (Is it safe to buy premium?)
        # For options, we want explosive moves, not slow grinds.
        
        atr = compute_atr(df, 14)
        atr_val = _safe_last(atr)
        atr_pct = atr_val / close
        
        rsi = compute_rsi(df["Close"], 14)
        rsi_val = _safe_last(rsi)
        
        ma20 = df["Close"].rolling(20).mean()
        ma20_val = _safe_last(ma20)
        
        # Scoring
        directional = 50.0
        if close > ma20_val:
            if 50 <= rsi_val <= 65: directional = 80.0 # Ideal for Calls
            elif rsi_val > 70: directional = 60.0 # Risk of pullback
            else: directional = 65.0
        else:
            if 35 <= rsi_val <= 50: directional = 20.0 # Ideal for Puts
            elif rsi_val < 30: directional = 40.0 # Risk of bounce
            else: directional = 35.0
            
        # Theta Risk Score (Volatility check)
        # If Vol is too low, buying options is dead money (Theta kills you).
        theta_risk = 50.0
        if atr_pct < 0.01: theta_risk = 20.0 # High Theta Risk (Price not moving)
        elif atr_pct > 0.03: theta_risk = 80.0 # Low Theta Risk (Price moving fast)
        
        comp = float(np.clip(directional * 0.6 + theta_risk * 0.4, 0, 100))
        
        if comp >= 75: direction, label = "UP", "BUY CALLS"
        elif comp >= 60: direction, label = "UP", "CALL SPREADS"
        elif comp <= 25: direction, label = "DOWN", "BUY PUTS"
        elif comp <= 40: direction, label = "DOWN", "PUT SPREADS"
        else: direction, label = "FLAT", "IRON CONDOR"
        
        factors = {
            "directional_edge": round(directional, 1),
            "theta_safety": round(theta_risk, 1),
            "composite_score": round(comp, 1)
        }
        
        # Targets (Implied Move)
        target = close + (atr_val * 1.5) if direction == "UP" else close - (atr_val * 1.5)
        stop = close - (atr_val * 0.8) if direction == "UP" else close + (atr_val * 0.8)
        
        return SignalResult(
            symbol=symbol.upper(),
            market="OPTIONS",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=factors["composite_score"],
            direction=direction,
            label=label,
            entry=round(close, 2),
            target=round(target, 2),
            stop=round(stop, 2),
            support=round(df.tail(20)["Low"].min(), 2),
            resistance=round(df.tail(20)["High"].max(), 2),
            time_frame="2-5 Days (Theta Optimal)",
            expected_bars_to_target=None,
            expected_time_to_target_hours=None,
            factors=factors,
            meta={"implied_vol_proxy": round(atr_pct*100, 2)}
        )