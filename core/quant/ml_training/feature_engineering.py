# core/quant/feature_engineering.py

import pandas as pd
import numpy as np

# Updated Feature List
FEATURES = [
    "ret_1", "ret_3", "log_ret", "body_size", "wick_ratio",
    "ema_diff", "rsi_14", "trend_strength",
    "atr_ratio", "bb_width", "ttm_squeeze",
    "vol_z", "whale_z", "flow_imbalance",
    "regime_tag", "efficiency_ratio", "volatility_slope"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()

    # 1. Price Action & Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)

    # Body & Wicks
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

    # Trend Strength (Binary)
    df['trend_strength'] = np.where(df['close'] > ema_20, 1, -1) * np.where(ema_20 > ema_50, 1, 0.5)

    # 3. Volatility (ATR)
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    df['atr_14'] = df['tr'].rolling(14).mean()
    # ATR Ratio: Current Volatility vs Average (Is it expanding?)
    df['atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(50).mean()

    # Volatility Slope (Are we heating up or cooling down?)
    df['volatility_slope'] = df['atr_14'].pct_change(3) * 100

    # Bollinger Bands
    std = df['close'].rolling(20).std()
    df['bb_width'] = (4 * std) / ema_20
    k_upper = ema_20 + (1.5 * df['atr_14'])
    df['ttm_squeeze'] = np.where((ema_20 + 2 * std) < k_upper, 1, 0)

    # 4. Volume Features
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

    vol_per_move = df['volume'] / (df['body_size'] + 0.0001)
    df['whale_z'] = (vol_per_move - vol_per_move.rolling(50).mean()) / vol_per_move.rolling(50).std()

    if 'taker_buy_base' in df.columns:
        df['flow_imbalance'] = df['taker_buy_base'] / (df['volume'] + 1e-9)
    else:
        df['flow_imbalance'] = 0.5

    # 5. NEW: Efficiency Ratio (Kaufman)
    # Measures "Path Efficiency". 1.0 = Straight line (Sniper). 0.0 = Chop.
    change = abs(df['close'] - df['close'].shift(10))
    volatility = df['tr'].rolling(10).sum()
    df['efficiency_ratio'] = change / (volatility + 1e-9)

    # Market Regime (1 = Trend Ready, 0 = Chop)
    # Require Expansion AND Efficiency
    df['regime_tag'] = np.where((df['atr_ratio'] > 1.0) & (df['efficiency_ratio'] > 0.3), 1, 0)

    return df.fillna(0)


# ------------------------------------------------------------
# DYNAMIC TARGETS (Strict)
# ------------------------------------------------------------
def generate_targets(df: pd.DataFrame, risk_reward=2.0, stop_mult=1.5, candles=12) -> pd.DataFrame:
    data = df.copy()
    close = data['close']
    atr = data['atr_14']

    # Dynamic Levels
    stop_dist = atr * stop_mult
    target_dist = stop_dist * risk_reward

    # Forward Look
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=candles)
    future_high = data['high'].rolling(window=indexer).max()
    future_low = data['low'].rolling(window=indexer).min()

    # Long Win: Hit TP (Entry+Target) BEFORE hitting SL (Entry-Stop)
    long_tp = close + target_dist
    long_sl = close - stop_dist
    data['target_long'] = ((future_high >= long_tp) & (future_low > long_sl)).astype(int)

    # Short Win: Hit TP (Entry-Target) BEFORE hitting SL (Entry+Stop)
    short_tp = close - target_dist
    short_sl = close + stop_dist
    data['target_short'] = ((future_low <= short_tp) & (future_high < short_sl)).astype(int)

    return data.dropna()