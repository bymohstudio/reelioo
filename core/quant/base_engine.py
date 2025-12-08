# core/quant/base_engine.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd




from dataclasses import dataclass

@dataclass
class SignalResult:
    signal: str
    edge: float
    ml: dict | None = None
    rule_score: float | None = None
    meta: dict | None = None
    raw: dict | None = None



# -------------------------------------------------------------------
# Shared result model for all quant engines
# -------------------------------------------------------------------


@dataclass
class QuantResult:
    """
    Unified result shape used by all engines and API layer.

    This MUST stay compatible with:
    - api_views.AnalyzeMarketView (asdict() → JSON)
    - BacktestEngine (expects direction, entry, target, stop, score, etc.)
    - Frontend UI (Confidence Score, Entry/Target/Stop).
    """

    symbol: str
    market_type: str          # EQUITY / FUTURES / OPTIONS / CRYPTO
    time_frame: str           # INTRADAY / SWING / LONG_TERM

    direction: str            # BUY / SELL / NEUTRAL
    score: float              # Final confidence score 0–100 (ML-dominant blend)
    ml_edge: float            # Pure ML edge 0–100 (prob * 100)
    raw_prob: float           # Raw ML probability (0–1)

    trend_score: float        # -100..+100 internal trend strength
    volatility_regime: str    # LOW / NORMAL / HIGH
    signal_quality: str       # A / B / C bucket for UX

    entry: float              # Suggested entry price (usually last close)
    target: float             # Suggested target
    stop: float               # Suggested stop-loss
    risk_reward: float        # |target-entry| / |entry-stop|

    extras: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# Core helpers shared by all engines
# -------------------------------------------------------------------


def compute_basic_ohlc_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes any OHLCV dataframe into:
        open, high, low, close, volume (lowercase)

    Handles:
    - yfinance multiindex columns
    - SmartAPI / WazirX / other sources with various naming.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Handle MultiIndex columns (yfinance)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[-1]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]

    # Map common aliases
    rename_map = {}
    for c in out.columns:
        lc = c.lower()
        if lc in ("o", "open"):
            rename_map[c] = "open"
        elif lc in ("h", "high"):
            rename_map[c] = "high"
        elif lc in ("l", "low"):
            rename_map[c] = "low"
        elif lc in ("c", "close", "price"):
            rename_map[c] = "close"
        elif "volume" in lc or lc == "vol":
            rename_map[c] = "volume"

    out = out.rename(columns=rename_map)

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    if not keep or "close" not in keep:
        return pd.DataFrame()

    out = out[keep].copy()

    # Coerce numeric
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Datetime index if possible
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass

    # Drop bad rows
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _simple_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic indicators used for TA sanity checks and feature snapshots.
    Keeps it light – this is not the same as ML feature engineering.
    """
    df = df.copy()
    if df.empty:
        return df

    close = df["close"]

    # Moving averages
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()

    # ATR
    if {"high", "low", "close"}.issubset(df.columns):
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
    else:
        df["atr_14"] = np.nan

    # RSI
    df["rsi_14"] = _simple_rsi(close, 14)

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)

    return df


def compute_trend_score(df: pd.DataFrame) -> float:
    """
    Returns a trend score in [-100, +100].

    +ve = bullish trend
    -ve = bearish trend
    near 0 = choppy / flat
    */
    """
    if df is None or len(df) < 50 or "close" not in df.columns:
        return 0.0

    close = df["close"]
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    last_close = close.iloc[-1]
    last_ma20 = ma20.iloc[-1]
    last_ma50 = ma50.iloc[-1]

    if any(np.isnan([last_close, last_ma20, last_ma50])):
        return 0.0

    # Distance from MA as % of price
    dist20 = (last_close - last_ma20) / last_close
    dist50 = (last_close - last_ma50) / last_close

    score = 0.0
    # Uptrend bias
    if last_close > last_ma20 > last_ma50:
        score += 60
    elif last_close > last_ma20:
        score += 40
    elif last_close > last_ma50:
        score += 20

    # Strength from distances
    score += 50 * np.tanh(3 * dist20)   # saturate
    score += 30 * np.tanh(3 * dist50)

    # clamp
    score = max(-100.0, min(100.0, score))
    return float(score)


def detect_volatility_regime(df: pd.DataFrame) -> str:
    """
    Classify vol regime using ATR / price.
    LOW / NORMAL / HIGH
    """
    if df is None or len(df) < 20 or "close" not in df.columns:
        return "UNKNOWN"

    if "atr_14" not in df.columns:
        df = add_core_indicators(df)

    last_close = df["close"].iloc[-1]
    atr = df["atr_14"].iloc[-1]
    if np.isnan(atr) or last_close <= 0:
        return "UNKNOWN"

    atr_pct = atr / last_close  # daily ATR as % of price

    if atr_pct < 0.0075:
        return "LOW"
    elif atr_pct < 0.02:
        return "NORMAL"
    else:
        return "HIGH"


def determine_signal_quality(score: float, ml_edge: float) -> str:
    """
    Bucket signal into A/B/C for UX.

    A → High confidence
    B → Medium
    C → Weak / avoid
    """
    if ml_edge >= 75 and score >= 70:
        return "A"
    if ml_edge >= 55 and score >= 55:
        return "B"
    return "C"


def clamp_score(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, val)))


def build_entry_target_stop(
    df: pd.DataFrame,
    direction: str,
    trade_style: str,
    atr_fallback: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Returns: entry, target, stop, risk_reward

    ATR-based sizing:
      - INTRADAY: 0.8 ATR stop, 1.4 ATR target
      - SWING:    1.3 ATR stop, 2.2 ATR target
      - LONG_TERM:1.8 ATR stop, 3.0 ATR target
    """

    last_row = df.iloc[-1]
    price = float(last_row["close"])

    atr = float(last_row.get("atr_14", np.nan))
    if np.isnan(atr) or atr <= 0:
        atr = atr_fallback * price * 0.01  # 1% of price as crude fallback

    style = trade_style.upper()

    if style == "INTRADAY":
        stop_mult = 0.8
        tgt_mult = 1.4
    elif style == "LONG_TERM":
        stop_mult = 1.8
        tgt_mult = 3.0
    else:  # SWING default
        stop_mult = 1.3
        tgt_mult = 2.2

    risk = stop_mult * atr
    reward = tgt_mult * atr

    if direction == "BUY":
        entry = price
        stop = max(0.01, price - risk)
        target = price + reward
    elif direction == "SELL":
        entry = price
        stop = price + risk
        target = max(0.01, price - reward)
    else:
        # NEUTRAL – use dummy but consistent values
        entry = price
        stop = price
        target = price
        return entry, target, stop, 0.0

    rr = 0.0
    denom = abs(entry - stop)
    if denom > 1e-9:
        rr = abs(target - entry) / denom

    return float(entry), float(target), float(stop), float(rr)
