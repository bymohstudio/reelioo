# core/quant/ml_training/feature_engineering.py

import pandas as pd
import numpy as np

# INSTITUTIONAL FEATURE SET (Fixed: Added volatility_slope)
FEATURES = [
    # Price Physics
    "ret_1", "log_ret", "body_size", "wick_ratio",
    # Smart Money Context
    "vwap_dist", "liq_sweep", "order_block",
    # Momentum & Trend
    "rsi_14", "ema_diff", "trend_strength",
    # Volatility & Energy
    "atr_ratio", "ttm_squeeze", "volatility_slope",  # <--- RESTORED
    # The "Red Pill" (Volume/Flow)
    "whale_z", "cvd_divergence", "flow_imbalance", "efficiency_ratio"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()

    # --- 1. BASIC PRICE ACTION ---
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['body_size'] = abs(df['close'] - df['open'])

    # Wick Ratio (The "Trap" Detector)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['wick_ratio'] = (upper_wick - lower_wick) / (df['body_size'] + 1e-9)

    # --- 2. INSTITUTIONAL ANCHORS (VWAP) ---
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vp = typical_price * df['volume']
    df['vwap'] = vp.rolling(24).sum() / df['volume'].rolling(24).sum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)

    # --- 3. LIQUIDITY SWEEPS (The "Stop Hunt") ---
    roll_high = df['high'].rolling(10).max().shift(1)
    roll_low = df['low'].rolling(10).min().shift(1)

    df['liq_sweep'] = 0
    # Bear Sweep: High > PrevHigh but Close < PrevHigh
    df.loc[(df['high'] > roll_high) & (df['close'] < roll_high), 'liq_sweep'] = -1
    # Bull Sweep: Low < PrevLow but Close > PrevLow
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
    df['atr_ratio'] = df['atr_14'] / df['atr_14'].rolling(50).mean()

    # --- FIX: Restore Volatility Slope Calculation ---
    df['volatility_slope'] = df['atr_14'].pct_change(3) * 100

    std = df['close'].rolling(20).std()
    k_upper = df['ema_20'] + (1.5 * df['atr_14'])
    df['ttm_squeeze'] = np.where((df['ema_20'] + 2 * std) < k_upper, 1, 0)

    # --- 6. ORDER FLOW ALPHA (The Lie Detector) ---
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)

    # Whale Activity (Effort vs Result)
    vol_per_move = df['volume'] / (df['body_size'] + 0.0001)
    df['whale_z'] = (vol_per_move - vol_per_move.rolling(50).mean()) / (vol_per_move.rolling(50).std() + 1e-9)

    # Order Block Potential
    df['order_block'] = np.where((df['vol_z'] > 1.5) & (df['body_size'] < df['atr_14'] * 0.3), 1, 0)

    if 'taker_base' in df.columns:
        taker_buy = df['taker_base']
        taker_sell = df['volume'] - taker_buy
        df['flow_imbalance'] = (taker_buy - taker_sell) / (df['volume'] + 1e-9)

        # CVD Slope
        df['cvd_slope'] = df['flow_imbalance'].rolling(3).sum()

        # CVD Divergence
        price_slope = df['close'].diff(3)
        cvd_slope_roc = df['cvd_slope'].diff(3)

        df['cvd_divergence'] = 0
        df.loc[(price_slope > 0) & (cvd_slope_roc < 0), 'cvd_divergence'] = -1
        df.loc[(price_slope < 0) & (cvd_slope_roc > 0), 'cvd_divergence'] = 1
    else:
        df['flow_imbalance'] = 0.0
        df['cvd_slope'] = 0.0
        df['cvd_divergence'] = 0.0

    # Efficiency
    change = abs(df['close'] - df['close'].shift(10))
    volatility = df['tr'].rolling(10).sum()
    df['efficiency_ratio'] = change / (volatility + 1e-9)

    return df.fillna(0)


def generate_targets(df: pd.DataFrame, risk_reward=2.0, stop_mult=1.5, candles=6) -> pd.DataFrame:
    data = df.copy()
    future_close = data['close'].shift(-candles)
    min_move = data['atr_14'] * 2.0
    data['target_long'] = np.where(
        (future_close > data['close'] + min_move) & (data['efficiency_ratio'] > 0.1),
        1, 0
    )
    data['target_short'] = np.where(
        (future_close < data['close'] - min_move) & (data['efficiency_ratio'] > 0.1),
        1, 0
    )
    return data.dropna()