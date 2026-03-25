import pandas as pd
import numpy as np

# These features match v36 REBALANCED logic
PHYSICS_FEATURES = [
    "force",
    "acceleration",
    "mass",
    "velocity",
    "energy_reserve",   # Hidden RSI
    "stretch_pct",      # Elasticity
    "structure_state",  # EMA alignment
    "is_trap",
    "vol_z_score",      # v36: Volume z-score
    "trend_consistency", # v36: Directional consistency
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v36 Feature Generator.
    Aligned with Rebalanced Confluence Engine.
    Adds: volume z-score, trend consistency, dual EMA structure.
    """
    if df.empty:
        return df

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # 1. PHYSICS VECTORS (Anti-Lag)
    # -------------------------
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    # ATR 14 (Sigma)
    sigma = tr.rolling(14).mean()
    df["atr_14"] = sigma

    # Velocity (Normalized Speed)
    df["velocity"] = close.diff() / (sigma + 1e-9)

    # Mass (Relative Volume)
    vol_mean = volume.rolling(20).mean()
    df["mass"] = volume / (vol_mean + 1e-9)

    # Force (Smoothed 2-period for Anti-Lag)
    raw_force = df["mass"] * df["velocity"]
    df["force"] = raw_force.rolling(2).mean()

    # Acceleration (Jerk) - The Derivative of Force
    df["acceleration"] = df["force"].diff(2)

    # -------------------------
    # 2. ENERGY GAUGE (Hidden RSI)
    # -------------------------
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["energy_reserve"] = 100 - (100 / (1 + rs))

    # -------------------------
    # 3. STRUCTURE & ELASTICITY
    # -------------------------
    # Dual EMA structure for v36
    eq_fast = close.ewm(span=34, adjust=False).mean()
    eq_slow = close.ewm(span=100, adjust=False).mean()

    # Elasticity (Stretch from fast EMA)
    df["stretch_pct"] = (close - eq_fast) / eq_fast

    # Structure State (v36: considers both EMAs)
    df["structure_state"] = np.where(
        (close > eq_fast) & (eq_fast > eq_slow),
        "BULLISH_ALIGNED",
        np.where(
            (close < eq_fast) & (eq_fast < eq_slow),
            "BEARISH_ALIGNED",
            np.where(
                close > eq_fast,
                "BULLISH_STRUCT",
                "BEARISH_STRUCT"
            )
        )
    )

    # -------------------------
    # 4. TRAP DETECTION (Wicks)
    # -------------------------
    open_p = df["open"]
    body = (close - open_p).abs()
    upper_wick = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_p], axis=1).min(axis=1) - low

    bull_trap = (df["force"] > 0) & (upper_wick > (body * 1.2))
    bear_trap = (df["force"] < 0) & (lower_wick > (body * 1.2))

    df["is_trap"] = bull_trap | bear_trap

    # -------------------------
    # 5. VOLUME Z-SCORE (v36 New)
    # -------------------------
    vol_std = volume.rolling(20).std()
    df["vol_z_score"] = (volume - vol_mean) / (vol_std + 1e-9)

    # -------------------------
    # 6. TREND CONSISTENCY (v36 New)
    # -------------------------
    # Rolling count of positive closes over last 6 bars
    pos_changes = (close.diff() > 0).astype(int)
    df["trend_consistency"] = pos_changes.rolling(6).sum()

    return df.fillna(0)