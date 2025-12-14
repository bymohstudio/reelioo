# core/quant/indicators.py

import pandas as pd
import numpy as np

# MUST match the features used in training (Part 1)
ML_FEATURES = ["ret_1", "vol_z", "trend_strength", "squeeze", "rsi", "atr_ratio"]


def compute_ml_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the exact feature set required by the XGBoost Model.
    """
    if raw_df.empty: return pd.DataFrame()
    df = raw_df.copy()

    # 1. Whale Z-Score (Volume)
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / vol_std

    # 2. Trend Strength (EMA Alignment)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['trend_strength'] = np.where(
        (df['close'] > df['ema_20']) & (df['ema_20'] > df['ema_50']), 1.0,
        np.where((df['close'] < df['ema_20']) & (df['ema_20'] < df['ema_50']), -1.0, 0.0)
    )

    # 3. Squeeze (Bollinger Width)
    std = df['close'].rolling(20).std()
    df['squeeze'] = (std * 2) / df['ema_20']

    # 4. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 5. ATR Ratio
    tr = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    atr = tr.rolling(14).mean()
    df['atr_ratio'] = tr / atr

    # 6. Returns
    df['ret_1'] = df['close'].pct_change()

    # Fill NaNs to avoid ML crashes
    df = df.fillna(0)

    return df