from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult, STYLE_CONFIG, _safe_last, _pick_style,
    compute_basic_ohlc_aliases, compute_atr, compute_rsi,
    compute_vol_regime, expected_bars_to_target, format_time_window, Direction
)

class EquityQuantEngine:
    MODEL_VERSION = "equity_global_v2.1"

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_basic_ohlc_aliases(df)
        close = df["Close"]
        df["ma20"] = close.rolling(20).mean()
        df["ma50"] = close.rolling(50).mean()
        df["ma200"] = close.rolling(200).mean()
        df["atr14"] = compute_atr(df, 14)
        df["rsi14"] = compute_rsi(close, 14)
        return df

    @classmethod
    def _trend_score(cls, df: pd.DataFrame) -> float:
        c = _safe_last(df["Close"])
        m20 = _safe_last(df["ma20"])
        m50 = _safe_last(df["ma50"])
        m200 = _safe_last(df["ma200"])
        
        score = 50.0
        # Perfect Bullish Alignment
        if c > m20 > m50 > m200: score = 90.0
        # Strong Bullish (Price > MAs)
        elif c > m20 and m20 > m50: score = 75.0
        # Recovery
        elif c > m20 and c > m50: score = 65.0
        # Bearish Breakdown
        elif c < m20 < m50 < m200: score = 10.0
        # Weakness
        elif c < m20 and c < m50: score = 30.0
        
        return float(np.clip(score, 0, 100))

    @classmethod
    def _momentum_score(cls, df: pd.DataFrame) -> float:
        rsi = _safe_last(df["rsi14"], 50)
        # Bullish Zones
        if 50 <= rsi <= 65: score = 75.0
        elif 40 <= rsi < 50: score = 60.0
        # Overbought
        elif rsi > 70: score = 40.0
        # Oversold Bounce
        elif rsi < 30: score = 35.0
        # Bearish
        elif rsi < 40: score = 25.0
        else: score = 50.0
        return float(np.clip(score, 0, 100))

    @classmethod
    def _structure_score(cls, df: pd.DataFrame) -> float:
        # Detects if price is making Higher Highs (Bullish Structure)
        recent = df.tail(20)
        highs = recent["High"]
        if highs.iloc[-1] >= highs.max(): return 80.0
        if highs.iloc[-1] >= highs.mean(): return 60.0
        return 40.0

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str | None = None) -> SignalResult:
        if df is None or df.empty: raise ValueError("Empty Dataframe")
        
        style = _pick_style(trade_style)
        df = cls._prepare(df)
        close = _safe_last(df["Close"])
        
        t = cls._trend_score(df)
        m = cls._momentum_score(df)
        s = cls._structure_score(df)
        
        # Composite Score
        comp = float(np.clip(t * 0.40 + m * 0.30 + s * 0.30, 0, 100))
        
        # Direction Logic
        if comp >= 75: direction, label = "UP", "STRONG BUY"
        elif comp >= 60: direction, label = "UP", "BUY"
        elif comp <= 25: direction, label = "DOWN", "STRONG SELL"
        elif comp <= 40: direction, label = "DOWN", "SELL"
        else: direction, label = "FLAT", "NEUTRAL"

        # Targets
        atr = _safe_last(df["atr14"], close * 0.02)
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
            entry = close
            target = close * 1.01
            stop = close * 0.99

        # Time Window
        dist = abs(target - entry)
        bars, hrs = expected_bars_to_target(dist, atr, cfg["bar_hours"])
        
        factors = {
            "trend_score": round(t, 1),
            "momentum_score": round(m, 1),
            "structure_score": round(s, 1),
            "composite_score": round(comp, 1)
        }
        
        return SignalResult(
            symbol=symbol.upper(),
            market="EQUITY",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=factors["composite_score"],
            direction=direction,
            label=label,
            entry=round(entry, 2),
            target=round(target, 2),
            stop=round(stop, 2),
            support=round(df.tail(50)["Low"].min(), 2),
            resistance=round(df.tail(50)["High"].max(), 2),
            time_frame=format_time_window(hrs),
            expected_bars_to_target=bars,
            expected_time_to_target_hours=hrs,
            factors=factors,
            meta={"atr": round(atr, 2)}
        )