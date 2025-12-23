# core/quant/ml_training/feature_engineering.py

import pandas as pd
import numpy as np

# INSTITUTIONAL FEATURE SET
FEATURES = [
    # Price
    "ret_1", "ret_3", "log_ret", "body_size", "wick_ratio",
    # Trend
    "ema_diff", "rsi_14", "trend_strength",
    # Volatility
    "atr_ratio", "bb_width", "ttm_squeeze", "volatility_slope",
    # Volume/Flow (The Alpha)
    "vol_z", "whale_z", "flow_imbalance", "cvd_slope", "efficiency_ratio"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()

    # --- 1. BASIC PRICE ACTION ---
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['body_size'] = abs(df['close'] - df['open'])

    # Wick Ratio (Rejection pressure)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_ratio'] = (upper_wick + lower_wick) / (df['body_size'] + 1e-9)

    # --- 2. MOMENTUM PHYSICS ---
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['close']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Trend Strength (Physical alignment)
    df['trend_strength'] = np.where(df['close'] > df['ema_20'], 1, -1) * np.where(df['ema_20'] > df['ema_50'], 1, 0.5)

    # --- 3. VOLATILITY REGIMES ---
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(50).mean()
    df['volatility_slope'] = df['atr_14'].pct_change(3) * 100

    # TTM Squeeze (Energy compression)
    std = df['close'].rolling(20).std()
    df['bb_width'] = (4 * std) / df['ema_20']
    k_upper = df['ema_20'] + (1.5 * df['atr_14'])
    df['ttm_squeeze'] = np.where((df['ema_20'] + 2 * std) < k_upper, 1, 0)

    # --- 4. ORDER FLOW & WHALES (THE REAL ALPHA) ---
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

    # Volume per pip (Effort vs Result)
    vol_per_move = df['volume'] / (df['body_size'] + 0.0001)
    df['whale_z'] = (vol_per_move - vol_per_move.rolling(50).mean()) / vol_per_move.rolling(50).std()

    # CVD / Flow Imbalance (Requires taker_base)
    if 'taker_base' in df.columns:
        # taker_base = Aggressive Buys
        # volume - taker_base = Aggressive Sells (approx)
        taker_buy = df['taker_base']
        taker_sell = df['volume'] - taker_buy

        # Raw Imbalance
        df['flow_imbalance'] = (taker_buy - taker_sell) / (df['volume'] + 1e-9)

        # CVD Slope (Is money entering NOW?)
        df['cvd_slope'] = df['flow_imbalance'].rolling(3).sum()
    else:
        df['flow_imbalance'] = 0.0
        df['cvd_slope'] = 0.0

    # --- 5. EFFICIENCY (Kaufman) ---
    change = abs(df['close'] - df['close'].shift(10))
    volatility = df['tr'].rolling(10).sum()
    df['efficiency_ratio'] = change / (volatility + 1e-9)

    # Regime Tag (Trend vs Chop) for context
    df['regime_tag'] = np.where((df['atr_ratio'] > 1.0) & (df['efficiency_ratio'] > 0.3), 1, 0)

    return df.fillna(0)


# ------------------------------------------------------------
# DYNAMIC TARGETS (INSTITUTIONAL REGIME LABELS)
# ------------------------------------------------------------
def generate_targets(df: pd.DataFrame, risk_reward=2.0, stop_mult=1.5, candles=6) -> pd.DataFrame:
    """
    Predict VOLATILITY EXPANSION.
    Target = 1 if price moves > 2 ATR in one direction (Explosion).
    """
    data = df.copy()

    # --- CRITICAL FIX: USE SHIFT INSTEAD OF ROLLING WINDOW ---
    # This gets the future close 'candles' periods ahead instantly without errors.
    future_close = data['close'].shift(-candles)

    # 2. Define "Significant Move" (Volatility Expansion)
    # If price moves 2x standard ATR, that is a Trend Day.
    min_move = data['atr_14'] * 2.0

    # 3. Label Explosions
    # TARGET LONG: Price Explodes Up AND Market is Efficient (Real move, not wick)
    data['target_long'] = np.where(
        (future_close > data['close'] + min_move) & (data['efficiency_ratio'] > 0.1),
        1, 0
    )

    # TARGET SHORT: Price Explodes Down AND Market is Efficient
    data['target_short'] = np.where(
        (future_close < data['close'] - min_move) & (data['efficiency_ratio'] > 0.1),
        1, 0
    )

    # 4. Remove the last 'candles' rows (they have NaN future_close)
    return data.dropna()