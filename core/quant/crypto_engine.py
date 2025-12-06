from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult, STYLE_CONFIG, _safe_last, _pick_style,
    compute_basic_ohlc_aliases, compute_atr, compute_rsi,
    compute_vol_regime, expected_bars_to_target, format_time_window, Direction
)

class CryptoQuantEngine:
    MODEL_VERSION = "crypto_binance_v2.0"

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_basic_ohlc_aliases(df)
        close = df["Close"]
        df["ma24"] = close.rolling(24).mean() # 24h crypto cycle
        df["ma72"] = close.rolling(72).mean()
        df["atr20"] = compute_atr(df, 20)
        df["rsi14"] = compute_rsi(close, 14)
        df["returns"] = close.pct_change()
        df["skew24"] = df["returns"].rolling(24).skew() # Tail risk
        return df

    @classmethod
    def _tail_risk_score(cls, df: pd.DataFrame) -> float:
        skew = _safe_last(df["skew24"], 0)
        score = 55.0
        # Negative skew = crash risk
        if skew < -0.5: score -= 15.0
        # Positive skew = pump potential
        if skew > 0.5: score += 10.0
        return float(np.clip(score, 20, 90))

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str | None = None) -> SignalResult:
        if df is None or df.empty: raise ValueError("Empty Dataframe")
        
        style = _pick_style(trade_style)
        df = cls._prepare(df)
        close = _safe_last(df["Close"])
        
        # Crypto-Specific Scoring
        rsi = _safe_last(df["rsi14"])
        
        # Momentum
        if 50 <= rsi <= 65: mom = 80.0
        elif rsi > 70: mom = 40.0 # Caution
        elif rsi < 30: mom = 30.0 
        else: mom = 50.0
        
        # Trend
        ma24 = _safe_last(df["ma24"])
        trend = 80.0 if close > ma24 else 30.0
        
        tail = cls._tail_risk_score(df)
        
        # Volatility Regime
        atr = _safe_last(df["atr20"], close*0.05)
        atr_pct = atr/close if close else 0.05
        vol_regime = compute_vol_regime(atr_pct)
        
        # Composite
        comp = float(np.clip(trend * 0.3 + mom * 0.35 + tail * 0.35, 0, 100))
        
        if comp >= 75: direction, label = "UP", "BULL RUN"
        elif comp >= 60: direction, label = "UP", "ACCUMULATE"
        elif comp <= 30: direction, label = "DOWN", "DUMP RISK"
        elif comp <= 45: direction, label = "DOWN", "BEARISH"
        else: direction, label = "FLAT", "CHOPPY"

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
            "momentum_score": round(mom, 1),
            "trend_score": round(trend, 1),
            "tail_risk_score": round(tail, 1),
            "composite_score": round(comp, 1)
        }

        return SignalResult(
            symbol=symbol.upper(),
            market="CRYPTO",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=factors["composite_score"],
            direction=direction,
            label=label,
            entry=round(entry, 4) if entry < 10 else round(entry, 2),
            target=round(target, 4) if target < 10 else round(target, 2),
            stop=round(stop, 4) if stop < 10 else round(stop, 2),
            support=round(df.tail(48)["Low"].min(), 2),
            resistance=round(df.tail(48)["High"].max(), 2),
            time_frame=format_time_window(hrs),
            expected_bars_to_target=bars,
            expected_time_to_target_hours=hrs,
            factors=factors,
            meta={"regime": vol_regime}
        )