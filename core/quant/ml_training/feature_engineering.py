# core/quant/ml_training/feature_engineering.py

import pandas as pd
import numpy as np

# FEATURES (Unchanged)
FEATURES = [
    "ret_1", "log_ret", "body_pct", "wick_ratio",
    "vwap_dist", "liq_sweep", "order_block",
    "rsi_14", "ema_diff", "trend_strength",
    "atr_pct", "ttm_squeeze", "volatility_slope",
    "whale_z", "cvd_divergence", "flow_imbalance", "efficiency_ratio"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes market data into institutional vectors.
    (This function remains EXACTLY as before to ensure compatibility)
    """
    if df.empty: return df
    df = df.copy()

    # --- 1. NORMALIZED PRICE ACTION ---
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['body_pct'] = abs(df['close'] - df['open']) / df['close']

    # Wick Ratio
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    df['wick_ratio'] = (upper_wick + lower_wick) / (total_range + 1e-9)

    # --- 2. INSTITUTIONAL ANCHORS ---
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vp = typical_price * df['volume']
    df['vwap'] = vp.rolling(24).sum() / df['volume'].rolling(24).sum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)

    # --- 3. LIQUIDITY SWEEPS ---
    roll_high = df['high'].rolling(10).max().shift(1)
    roll_low = df['low'].rolling(10).min().shift(1)
    df['liq_sweep'] = 0
    df.loc[(df['high'] > roll_high) & (df['close'] < roll_high), 'liq_sweep'] = -1
    df.loc[(df['low'] < roll_low) & (df['close'] > roll_low), 'liq_sweep'] = 1

    # --- 4. MOMENTUM & TREND ---
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['close']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['trend_strength'] = np.where(df['close'] > df['ema_20'], 1, -1)

    # --- 5. VOLATILITY ENERGY ---
    df['tr'] = np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)))
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']
    df['volatility_slope'] = df['atr_14'].pct_change(3) * 100

    std = df['close'].rolling(20).std()
    k_upper = df['ema_20'] + (1.5 * df['atr_14'])
    df['ttm_squeeze'] = np.where((df['ema_20'] + 2 * std) < k_upper, 1, 0)

    # --- 6. ORDER FLOW ALPHA ---
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

    vol_per_move = df['volume'] / (df['body_pct'] + 1e-9)
    df['whale_z'] = (vol_per_move - vol_per_move.rolling(50).mean()) / (vol_per_move.rolling(50).std() + 1e-9)

    df['order_block'] = np.where((df['vol_z'] > 1.5) & (df['body_pct'] < df['atr_pct'] * 0.3), 1, 0)

    if 'taker_base' in df.columns:
        taker_buy = df['taker_base']
        taker_sell = df['volume'] - taker_buy
        df['flow_imbalance'] = (taker_buy - taker_sell) / (df['volume'] + 1e-9)
        df['cvd_slope'] = df['flow_imbalance'].rolling(3).sum()

        price_slope = df['close'].diff(3)
        cvd_slope_roc = df['cvd_slope'].diff(3)

        df['cvd_divergence'] = 0
        df.loc[(price_slope > 0) & (cvd_slope_roc < 0), 'cvd_divergence'] = -1
        df.loc[(price_slope < 0) & (cvd_slope_roc > 0), 'cvd_divergence'] = 1
    else:
        df['flow_imbalance'] = 0.0
        df['cvd_slope'] = 0.0
        df['cvd_divergence'] = 0.0

    change = abs(df['close'] - df['close'].shift(10))
    volatility = df['tr'].rolling(10).sum()
    df['efficiency_ratio'] = change / (volatility + 1e-9)

    return df.fillna(0)


# --- ASYMMETRIC PHYSICS (THE FIX) ---
def generate_targets(df: pd.DataFrame, risk_reward=2.0, stop_mult=1.0, candles=24) -> pd.DataFrame:
    """
    Applies asymmetric logic for Crypto Markets:
    - LONGS: Look for Trends (24h horizon, 2.0x ATR)
    - SHORTS: Look for Dumps (12h horizon, 1.5x ATR)
    """
    data = df.copy()
    atr = data['atr_14']
    close = data['close']

    # --- LONG LOGIC (Trend Following) ---
    # Horizon: 24 Candles (Keep original setting)
    future_close_long = close.shift(-candles)
    target_long = atr * risk_reward  # 2.0 ATR

    # Must move 2.0 ATR up
    data['target_long'] = np.where(
        future_close_long > close + target_long,
        1, 0
    )

    # --- SHORT LOGIC (Panic Catching) ---
    # Horizon: 12 Candles (Faster)
    # Logic: Dumps happen fast. If it hasn't dumped in 12h, it's not a dump.
    future_close_short = close.shift(-12)
    target_short = atr * 1.5  # 1.5 ATR (Easier target for shorts)

    # Must move 1.5 ATR down
    data['target_short'] = np.where(
        future_close_short < close - target_short,
        1, 0
    )

    return data.dropna()