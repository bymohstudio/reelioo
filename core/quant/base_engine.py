# core/quant/base_engine.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class QuantResult:
    symbol: str = ""
    market_type: str = ""
    time_frame: str = ""
    direction: str = "NEUTRAL"
    score: float = 0.0
    ml_edge: float = 0.0
    raw_prob: float = 0.0
    trend_score: float = 0.0
    volatility_regime: str = "UNKNOWN"
    signal_quality: str = "C"
    entry: Optional[float] = None
    target: Optional[float] = None
    stop: Optional[float] = None
    risk_reward: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        return {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "time_frame": self.time_frame,
            "direction": self.direction,
            "score": float(self.score),
            "ml_edge": float(self.ml_edge),
            "trend_score": float(self.trend_score),
            "volatility_regime": self.volatility_regime,
            "entry": self.entry,
            "target": self.target,
            "stop": self.stop,
            "extras": self.extras or {},
        }


def compute_basic_ohlc_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes any OHLCV dataframe."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # Normalize columns to lowercase first
    out.columns = [str(c).lower() for c in out.columns]

    # Map aliases strictly
    rename_map = {}
    for c in out.columns:
        if c in ["open", "high", "low", "close", "volume"]: continue
        if "open" in c:
            rename_map[c] = "open"
        elif "high" in c:
            rename_map[c] = "high"
        elif "low" in c:
            rename_map[c] = "low"
        elif "close" in c or "price" in c:
            rename_map[c] = "close"
        elif "vol" in c:
            rename_map[c] = "volume"

    out = out.rename(columns=rename_map)

    # CRITICAL: Only require Price data. Volume is optional (for Indices).
    req = ["open", "high", "low", "close"]
    if not all(c in out.columns for c in req):
        return pd.DataFrame()  # Missing core data

    # Numeric coercion
    for c in req:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    # FIX: Drop only if PRICES are missing. Keep rows where Vol is NaN.
    return out.dropna(subset=req)


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    close = df["close"]
    # Simple MAs
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()

    # ATR
    if "high" in df.columns and "low" in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - close.shift(1)).abs()
        tr3 = (df["low"] - close.shift(1)).abs()
        df["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    return df


def compute_trend_score(df: pd.DataFrame) -> float:
    if df.empty or "ma_20" not in df.columns: return 0.0
    last = df.iloc[-1]
    score = 0
    if last["close"] > last["ma_20"]:
        score += 20
    else:
        score -= 20
    if last["close"] > last["ma_50"]:
        score += 20
    else:
        score -= 20
    if last["ma_20"] > last["ma_50"]:
        score += 10
    else:
        score -= 10
    return float(score)


def detect_volatility_regime(df: pd.DataFrame) -> str:
    if df.empty or "atr_14" not in df.columns: return "NORMAL"
    last = df.iloc[-1]
    atr_pct = (last["atr_14"] / last["close"]) * 100
    return "HIGH" if atr_pct > 1.5 else "NORMAL"


def determine_signal_quality(score: float, ml_edge: float) -> str:
    if score > 70 and ml_edge > 60: return "A+"
    if score > 60: return "A"
    if score > 50: return "B"
    return "C"


def clamp_score(val: float) -> float:
    return max(0.0, min(100.0, float(val)))


def build_entry_target_stop(df, direction, trade_style, atr_fallback=1.0):
    if df.empty: return 0, 0, 0, 0
    price = df["close"].iloc[-1]
    atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else (price * 0.01)
    if pd.isna(atr) or atr <= 0: atr = price * 0.01

    if trade_style == "INTRADAY":
        stop_mult, tgt_mult = 0.5, 1.2
    else:
        stop_mult, tgt_mult = 1.5, 2.5

    if direction in ["BULLISH", "BUY"]:
        entry = price
        stop = price - (atr * stop_mult)
        target = price + (atr * tgt_mult)
    elif direction in ["BEARISH", "SELL"]:
        entry = price
        stop = price + (atr * stop_mult)
        target = price - (atr * tgt_mult)
    else:
        entry, stop, target = 0.0, 0.0, 0.0

    return entry, target, stop, 0.0