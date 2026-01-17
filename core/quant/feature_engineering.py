import pandas as pd
import numpy as np

# These features match v31 TITANIUM logic
PHYSICS_FEATURES = [
    "force",
    "acceleration",
    "mass",
    "velocity",
    "energy_reserve",  # Hidden RSI
    "stretch_pct",  # Elasticity
    "structure_state",  # EMA 50 Alignment
    "is_trap"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v31 TITANIUM Feature Generator.
    Aligned with Anti-Lag, Elasticity, and Acceleration logic.
    """
    if df.empty:
        return df

    # Optimization: Avoid deep copy if not needed, but safe to keep for data integrity
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # 1. PHYSICS VECTORS (v31 Anti-Lag)
    # -------------------------
    # True Range & Sigma
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
    # 0-100 Scale
    df["energy_reserve"] = 100 - (100 / (1 + rs))

    # -------------------------
    # 3. STRUCTURE & ELASTICITY
    # -------------------------
    # Equilibrium (EMA 50)
    eq = close.ewm(span=50, adjust=False).mean()

    # Elasticity (Stretch from Mean)
    df["stretch_pct"] = (close - eq) / eq

    # Structure State (Bullish/Bearish based on EMA)
    df["structure_state"] = np.where(
        close > eq,
        "BULLISH_STRUCT",
        "BEARISH_STRUCT"
    )

    # -------------------------
    # 4. TRAP DETECTION (Wicks)
    # -------------------------
    open_p = df["open"]
    body = (close - open_p).abs()
    upper_wick = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_p], axis=1).min(axis=1) - low

    # Trap Logic: High Force vs Big Wick
    bull_trap = (df["force"] > 0) & (upper_wick > (body * 1.2))
    bear_trap = (df["force"] < 0) & (lower_wick > (body * 1.2))

    df["is_trap"] = bull_trap | bear_trap

    return df.fillna(0)