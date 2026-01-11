import pandas as pd
import numpy as np

# These features are DESCRIPTIVE ONLY
# They do NOT decide trades (The Engine calculates its own decision logic)

PHYSICS_FEATURES = [
    "signed_ke",
    "ke_decay",
    "mass",
    "velocity",
    "atr_14",
    "atr_pct",
    "structure_state",
    "liquidity_state",
    "event_risk",
    "volatility_compression"
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-aligned feature generator (v21).
    Matches CryptoQuantEngine & CryptoBacktestEngine logic.

    PURPOSE:
    - Pre-calculation for Backtesting
    - Visuals for UI
    """
    if df.empty:
        return df

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # 1. TRUE RANGE / ATR (Match v21)
    # -------------------------
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # -------------------------
    # 2. MASS (Participation)
    # -------------------------
    vol_avg = volume.rolling(20).mean()
    df["mass"] = volume / (vol_avg + 1e-9)

    # -------------------------
    # 3. VELOCITY (Directional)
    # -------------------------
    # Normalized by ATR to be asset-agnostic
    df["velocity"] = close.diff() / (df["atr_14"] + 1e-9)

    # -------------------------
    # 4. SIGNED KINETIC ENERGY (v21 Logic)
    # -------------------------
    # Removed the 0.5 scalar to match Engine logic exactly
    df["signed_ke"] = df["mass"] * df["velocity"]

    # -------------------------
    # 5. ENERGY DECAY (Slope)
    # -------------------------
    # CHANGED: Now returns a slope (float), not a boolean
    df["ke_decay"] = df["signed_ke"].diff(3)

    # -------------------------
    # 6. STRUCTURE STATE (20-Period Lookback)
    # -------------------------
    # CHANGED: Matches Engine's 20-candle structure check
    roll_high = high.rolling(20).max().shift(1)
    roll_low = low.rolling(20).min().shift(1)

    hh = high > roll_high
    hl = low > roll_low
    lh = high < roll_high
    ll = low < roll_low

    df["structure_state"] = np.select(
        [hh & hl, lh & ll],
        ["UPTREND", "DOWNTREND"],
        default="RANGE"
    )

    # -------------------------
    # 7. LIQUIDITY STATE (Fakeout Detection)
    # -------------------------
    # Using dynamic ATR threshold from v21 (2.5x ATR)
    candle_range = high - low
    is_wide = candle_range > (2.5 * df["atr_14"])
    vol_intensity = df["mass"]

    df["liquidity_state"] = np.where(
        is_wide & (vol_intensity < 0.8),
        "FAKEOUT",
        "CLEAN"
    )

    # -------------------------
    # 8. EVENT RISK (Descriptive)
    # -------------------------
    velocity_spike = df["velocity"].abs() > 2.5
    df["event_risk"] = velocity_spike

    # -------------------------
    # 9. VOLATILITY COMPRESSION
    # -------------------------
    long_vol = tr.rolling(100).mean()
    df["volatility_compression"] = df["atr_14"] / (long_vol + 1e-9)

    return df.fillna(0)