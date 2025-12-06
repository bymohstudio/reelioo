from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult, STYLE_CONFIG, _safe_last, _pick_style,
    compute_basic_ohlc_aliases, compute_atr, compute_rsi,
    expected_bars_to_target, format_time_window, Direction
)

class ForexQuantEngine:
    MODEL_VERSION = "forex_v1.0"

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str | None = None) -> SignalResult:
        if df is None or df.empty: raise ValueError("Empty Data")
        style = _pick_style(trade_style)
        df = compute_basic_ohlc_aliases(df)
        close = _safe_last(df["Close"])
        
        # Forex typically mean reverts on shorter frames
        rsi = _safe_last(compute_rsi(df["Close"], 14))
        bb_mean = _safe_last(df["Close"].rolling(20).mean())
        
        # Scoring
        score = 50.0
        if close > bb_mean:
            if rsi > 70: score = 40.0 # Reversion Short
            elif rsi > 50: score = 70.0 # Trend Continuation Long
        else:
            if rsi < 30: score = 60.0 # Reversion Long
            elif rsi < 50: score = 30.0 # Trend Continuation Short
            
        if score >= 60: direction, label = "UP", "LONG"
        elif score <= 40: direction, label = "DOWN", "SHORT"
        else: direction, label = "FLAT", "WAIT"
        
        # Forex Pips Calculation
        atr = _safe_last(compute_atr(df, 14))
        if direction == "UP":
            target = close + atr * 2
            stop = close - atr
        elif direction == "DOWN":
            target = close - atr * 2
            stop = close + atr
        else:
            target, stop = close, close
            
        factors = {"rsi": round(rsi, 1), "composite_score": score}
        
        return SignalResult(
            symbol=symbol.upper(),
            market="FOREX",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=score,
            direction=direction,
            label=label,
            entry=round(close, 5),
            target=round(target, 5),
            stop=round(stop, 5),
            support=round(df.tail(50)["Low"].min(), 5),
            resistance=round(df.tail(50)["High"].max(), 5),
            time_frame="Intraday/Swing",
            expected_bars_to_target=None,
            expected_time_to_target_hours=None,
            factors=factors,
            meta={}
        )