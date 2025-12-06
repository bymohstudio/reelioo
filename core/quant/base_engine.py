from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

import numpy as np
import pandas as pd

TradeStyle = Literal["INTRADAY", "SWING", "LONG_TERM"]
Direction = Literal["UP", "DOWN", "FLAT"]

STYLE_CONFIG: dict[TradeStyle, dict[str, Any]] = {
    "INTRADAY": {"target_atr": 0.8, "stop_atr": 0.5, "bar_hours": 0.25},
    "SWING": {"target_atr": 1.5, "stop_atr": 1.0, "bar_hours": 4.0},
    "LONG_TERM": {"target_atr": 3.0, "stop_atr": 1.8, "bar_hours": 24.0},
}

@dataclass
class SignalResult:
    symbol: str
    market: str
    style: TradeStyle
    model_version: str

    score: float
    direction: Direction
    label: str

    entry: float
    target: float
    stop: float

    support: Optional[float]
    resistance: Optional[float]

    time_frame: str
    expected_bars_to_target: Optional[float]
    expected_time_to_target_hours: Optional[float]

    factors: Dict[str, float]
    meta: Dict[str, Any]

def _safe_last(series: pd.Series, default=None) -> float:
    try:
        val = float(series.dropna().iloc[-1])
        return 0.0 if np.isnan(val) else val
    except Exception:
        return default or 0.0

def _pick_style(style: Optional[str]) -> TradeStyle:
    if not style:
        return "SWING"
    s = style.strip().upper()
    if s in ("INTRADAY", "SCALP", "DAY"):
        return "INTRADAY"
    if s in ("LONG", "LONGTERM", "POSITION", "LONG_TERM"):
        return "LONG_TERM"
    return "SWING"

def compute_basic_ohlc_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
        "adj close": "Close"
    }
    # Clean column names
    df.columns = [c.lower() for c in df.columns]
    
    # Remap back to Title Case for internal use
    rev_map = {k: v for k, v in mapping.items()}
    for col in df.columns:
        if col in rev_map:
            df.rename(columns={col: rev_map[col]}, inplace=True)
            
    # Ensure essentials
    for req in ["Open", "High", "Low", "Close"]:
        if req not in df.columns and "Close" in df.columns:
            df[req] = df["Close"]
            
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_vol_regime(atr_pct: float) -> str:
    if atr_pct < 0.015:
        return "LOW_VOL_MEAN_REVERT"
    if atr_pct < 0.04:
        return "HEALTHY_TREND_FRIENDLY"
    if atr_pct < 0.08:
        return "HIGH_ENERGY_BREAKOUT"
    return "EXTREME_RISK"

def expected_bars_to_target(target_abs: float, atr: float, bar_hours: float):
    if atr <= 0 or target_abs <= 0:
        return 0.0, 0.0
    n_atr = target_abs / atr
    bars = n_atr / 0.6  # approx 0.6 ATR per bar net movement in trend
    hours = bars * bar_hours
    return float(bars), float(hours)

def format_time_window(hours: Optional[float]) -> str:
    if not hours: return "N/A"
    if hours <= 8: return "Same-day move"
    if hours <= 72: return "2–3 sessions"
    if hours <= 24 * 14: return "1–2 weeks"
    return "Multi-week window"