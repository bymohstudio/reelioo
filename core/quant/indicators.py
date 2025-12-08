"""
Indicators used during LIVE runtime.
This version EXACTLY MATCHES the ML-training feature set.
"""

import pandas as pd
import numpy as np


# ----------------------------------------------------------
# BASIC HELPERS
# ----------------------------------------------------------

def _safe_shift(s, n):
    try:
        return s.shift(n)
    except Exception:
        return pd.Series([np.nan] * len(s))


# ----------------------------------------------------------
# INDICATOR BUILDER (MATCHES ML FEATURES EXACTLY)
# ----------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    Returns *one* dict = the last row of indicators,
    matching ML-training feature names, EXACTLY.

    Required ML features:
        open, high, low, close, volume,
        ret_1, ret_3, ret_5, ret_10,
        vol_10, vol_20,
        ma_10, ma_20, ma_50,
        close_over_ma20, close_over_ma50,
        atr_14, rsi_14,
        body_to_range, upper_wick, lower_wick,
        vol_zscore_20
    """

    if df is None or len(df) < 50:
        return {}

    df = df.copy()

    # --------------------------------------
    # Basic OHLC
    # --------------------------------------
    # Ensure columns exist and are numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # --------------------------------------
    # Returns
    # --------------------------------------
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # --------------------------------------
    # Volume rolling windows
    # --------------------------------------
    df["vol_10"] = df["volume"].rolling(10).mean()
    df["vol_20"] = df["volume"].rolling(20).mean()

    # --------------------------------------
    # Moving averages
    # --------------------------------------
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # --------------------------------------
    # Price relative to MAs
    # --------------------------------------
    df["close_over_ma20"] = df["close"] / (df["ma_20"] + 1e-9)
    df["close_over_ma50"] = df["close"] / (df["ma_50"] + 1e-9)

    # --------------------------------------
    # ATR
    # --------------------------------------
    df["tr1"] = (df["high"] - df["low"]).abs()
    df["tr2"] = (df["high"] - df["close"].shift(1)).abs()
    df["tr3"] = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # --------------------------------------
    # RSI
    # --------------------------------------
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --------------------------------------
    # Candle structure
    # --------------------------------------
    df["body_to_range"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # --------------------------------------
    # Volume z-score
    # --------------------------------------
    vol_mean_20 = df["volume"].rolling(20).mean()
    vol_std_20 = df["volume"].rolling(20).std()
    df["vol_zscore_20"] = (df["volume"] - vol_mean_20) / (vol_std_20 + 1e-9)

    # --------------------------------------
    # Extract last row as dict
    # --------------------------------------
    last = df.iloc[-1]

    # REQUIRED: These match the XGBoost model feature names exactly
    FEATURES = [
        "open", "high", "low", "close", "volume",
        "ret_1", "ret_3", "ret_5", "ret_10",
        "vol_10", "vol_20",
        "ma_10", "ma_20", "ma_50",
        "close_over_ma20", "close_over_ma50",
        "atr_14", "rsi_14",
        "body_to_range", "upper_wick", "lower_wick",
        "vol_zscore_20"
    ]

    # Safely extract features, defaulting to 0.0 if NaN/Infinite
    result = {}
    for f in FEATURES:
        val = last.get(f, 0.0)
        # Handle NaN or Inf
        if pd.isna(val) or np.isinf(val):
            val = 0.0
        result[f] = float(val)

    return result