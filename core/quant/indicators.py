# core/quant/indicators.py

import pandas as pd
import numpy as np

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


def compute_all_indicators(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 20:
        return {}

    df = df.copy()
    close = df["close"]

    # Returns
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)

    # Volatility
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # MAs
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()

    df["close_over_ma20"] = (close / df["ma_20"]) - 1
    df["close_over_ma50"] = (close / df["ma_50"]) - 1

    # ATR & Candle
    if "high" in df.columns and "low" in df.columns:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - close.shift(1)).abs()
        tr3 = (df["low"] - close.shift(1)).abs()
        df["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_to_range"] = (close - df["open"]).abs() / rng
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume Z (Safe Handling for Indices)
    if "volume" in df.columns:
        # Check if volume is all NaN or Zero (Indices)
        if df["volume"].sum() == 0 or df["volume"].isna().all():
            df["vol_zscore_20"] = 0.0
            df["volume"] = 0.0
        else:
            df["vol_zscore_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    else:
        df["vol_zscore_20"] = 0.0
        df["volume"] = 0.0

    try:
        last_row = df.iloc[-1]
        result = {}
        for k in FEATURES:
            # Safely get float, handle NaN -> 0.0
            val = float(last_row.get(k, 0.0))
            if np.isnan(val) or np.isinf(val): val = 0.0
            result[k] = val
        return result
    except Exception:
        return {}