from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult, STYLE_CONFIG, _safe_last, _pick_style,
    compute_basic_ohlc_aliases, compute_atr, compute_rsi,
    expected_bars_to_target, format_time_window, Direction
)

class FuturesQuantEngine:
    MODEL_VERSION = "futures_global_v2.0"

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_basic_ohlc_aliases(df)
        close = df["Close"]
        df["ma21"] = close.rolling(21).mean()
        df["ma55"] = close.rolling(55).mean()
        df["atr10"] = compute_atr(df, 10)
        df["rsi7"] = compute_rsi(close, 7) # Faster RSI for futures
        return df

    @classmethod
    def _leverage_risk_score(cls, df: pd.DataFrame) -> float:
        # Measures volatility relative to price. High vol = High leverage risk.
        c = _safe_last(df["Close"])
        atr = _safe_last(df["atr10"])
        atr_pct = atr / c if c > 0 else 0.0
        
        score = 80.0
        if atr_pct > 0.02: score -= 20 # 2% daily move is huge for index futures
        if atr_pct > 0.04: score -= 30
        return float(np.clip(score, 10, 100))

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str | None = None) -> SignalResult:
        if df is None or df.empty: raise ValueError("Empty Data")
        
        style = _pick_style(trade_style)
        df = cls._prepare(df)
        close = _safe_last(df["Close"])
        
        # Trend
        m21 = _safe_last(df["ma21"])
        m55 = _safe_last(df["ma55"])
        if close > m21 > m55: trend = 85.0
        elif close < m21 < m55: trend = 15.0
        else: trend = 50.0
        
        # Momentum
        rsi = _safe_last(df["rsi7"])
        if 40 < rsi < 60: mom = 50.0
        elif rsi >= 60: mom = 75.0
        elif rsi <= 40: mom = 25.0
        else: mom = 50.0
        
        lev = cls._leverage_risk_score(df)
        
        comp = float(np.clip(trend * 0.4 + mom * 0.4 + lev * 0.2, 0, 100))
        
        if comp >= 70: direction, label = "UP", "LONG BIAS"
        elif comp <= 30: direction, label = "DOWN", "SHORT BIAS"
        else: direction, label = "FLAT", "NEUTRAL"
        
        # Targets
        atr = _safe_last(df["atr10"], close*0.01)
        cfg = STYLE_CONFIG[style]
        
        if direction == "UP":
            entry = close
            target = close + (atr * cfg["target_atr"])
            stop = close - (atr * cfg["stop_atr"])
        elif direction == "DOWN":
            entry = close
            target = close - (atr * cfg["target_atr"])
            stop = close + (atr * cfg["stop_atr"])
        else:
            entry, target, stop = close, close, close

        dist = abs(target - entry)
        bars, hrs = expected_bars_to_target(dist, atr, cfg["bar_hours"])
        
        factors = {
            "trend_score": round(trend, 1),
            "momentum_score": round(mom, 1),
            "leverage_risk_score": round(lev, 1),
            "composite_score": round(comp, 1)
        }
        
        return SignalResult(
            symbol=symbol.upper(),
            market="FUTURES",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=factors["composite_score"],
            direction=direction,
            label=label,
            entry=round(entry, 2),
            target=round(target, 2),
            stop=round(stop, 2),
            support=round(df.tail(30)["Low"].min(), 2),
            resistance=round(df.tail(30)["High"].max(), 2),
            time_frame=format_time_window(hrs),
            expected_bars_to_target=bars,
            expected_time_to_target_hours=hrs,
            factors=factors,
            meta={}
        )