# core/quant/feature_engineering.py

import pandas as pd
import numpy as np

# Feature List (Must match Model Trainer)
FEATURES = [
    "ret_1", "ret_3", "log_ret", "body_size", "wick_ratio",
    "ema_diff", "rsi_14", "trend_strength",
    "atr_ratio", "bb_width", "ttm_squeeze",
    "vol_z", "whale_z", "flow_imbalance",
    "regime_tag", "funding_trend"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()

    # 1. Returns & Price Action
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)

    df['body_size'] = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_ratio'] = (upper_wick + lower_wick) / (df['body_size'] + 1e-9)

    # 2. Trend & Momentum
    ema_20 = df['close'].ewm(span=20).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    df['ema_diff'] = (ema_20 - ema_50) / df['close']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Simple Trend Strength (0-1)
    df['trend_strength'] = np.where(df['close'] > ema_20, 1, -1) * np.where(ema_20 > ema_50, 1, 0.5)

    # 3. Volatility
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(50).mean()

    # Bollinger Width
    std = df['close'].rolling(20).std()
    df['bb_width'] = (4 * std) / ema_20

    # Squeeze (Bollinger inside Keltner)
    k_upper = ema_20 + (1.5 * df['atr_14'])
    df['ttm_squeeze'] = np.where((ema_20 + 2 * std) < k_upper, 1, 0)

    # 4. Volume & Flow (Whale Detection)
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

    # Whale Z: Abnormally high volume relative to candle size
    vol_per_move = df['volume'] / (df['body_size'] + 0.0001)
    df['whale_z'] = (vol_per_move - vol_per_move.rolling(50).mean()) / vol_per_move.rolling(50).std()

    # Flow Imbalance (Taker Buy vs Volume)
    if 'taker_buy_base' in df.columns:
        df['flow_imbalance'] = df['taker_buy_base'] / (df['volume'] + 1e-9)
    else:
        df['flow_imbalance'] = 0.5

    # 5. Market Structure
    # 1 = Trend, 0 = Chop
    df['regime_tag'] = np.where(df['atr_ratio'] > 1.0, 1, 0)

    # Funding Trend
    if 'fundingRate' in df.columns:
        df['funding_trend'] = df['fundingRate'] * 1000  # Scale up
    else:
        df['funding_trend'] = 0

    return df.fillna(0)