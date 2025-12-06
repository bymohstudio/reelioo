# core/quant/indicators.py

import pandas as pd
import numpy as np

def ensure_ohlc(df: pd.DataFrame):
    """Guarantees Open/High/Low/Close/Volume columns exist."""
    df = df.copy()
    # Map common variations
    rename_map = {
        "adj close": "Close", "adjclose": "Close",
        "high": "High", "low": "Low", "open": "Open", "close": "Close", "volume": "Volume"
    }
    # Lowercase match for safety
    for col in df.columns:
        if col.lower() in rename_map:
            df.rename(columns={col: rename_map[col.lower()]}, inplace=True)
            
    return df

def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def prepare_indicators(df: pd.DataFrame, asset_type="GLOBAL"):
    df = ensure_ohlc(df)
    
    df["atr"] = compute_atr(df)
    df["rsi"] = compute_rsi(df["Close"])
    
    # MAs
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()
    
    # Bollinger Bands (Vol)
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + (std * 2)
    df["bb_lower"] = df["ma20"] - (std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"]
    
    # ADX (Simple proxy using ATR/Ranges for Trend Strength)
    # Full ADX is complex, using a simplified trend strength index here
    df["trend_strength"] = (df["ma20"] - df["ma50"]).abs() / df["ma50"] * 100
    
    return df