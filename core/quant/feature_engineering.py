# core/quant/feature_engineering.py
import pandas as pd
import numpy as np

# These features are DESCRIPTIVE ONLY
# They do NOT decide trades

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
    Physics-aligned feature generator.
    Compatible with CryptoQuantEngine v19 / v20.

    PURPOSE:
    - Diagnostics
    - Explainability
    - Research
    - UI context

    NOT USED FOR DECISION MAKING.
    """
    if df.empty:
        return df

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # TRUE RANGE / ATR
    # -------------------------
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / close

    # -------------------------
    # MASS (Participation)
    # -------------------------
    vol_avg = volume.rolling(20).mean()
    df["mass"] = volume / (vol_avg + 1e-6)

    # -------------------------
    # VELOCITY (Directional)
    # -------------------------
    df["velocity"] = close.diff() / (df["atr_14"] + 1e-6)

    # -------------------------
    # SIGNED KINETIC ENERGY
    # -------------------------
    ke = 0.5 * df["mass"] * df["velocity"].abs()
    df["signed_ke"] = ke * np.sign(df["velocity"])

    # -------------------------
    # ENERGY DECAY (Exhaustion)
    # -------------------------
    ke_peak = df["signed_ke"].abs().rolling(5).max()
    df["ke_decay"] = df["signed_ke"].abs() < (ke_peak * 0.65)

    # -------------------------
    # STRUCTURE STATE (DESCRIPTIVE)
    # -------------------------
    hh = high > high.shift(1)
    hl = low > low.shift(1)
    lh = high < high.shift(1)
    ll = low < low.shift(1)

    df["structure_state"] = np.select(
        [
            hh & hl,
            lh & ll
        ],
        [
            "UPTREND",
            "DOWNTREND"
        ],
        default="RANGE"
    )

    # -------------------------
    # LIQUIDITY STATE
    # -------------------------
    wick_up = high - np.maximum(df["open"], close)
    wick_down = np.minimum(df["open"], close) - low
    wick_ratio = (wick_up + wick_down) / (tr + 1e-6)

    df["liquidity_state"] = np.where(
        wick_ratio > 0.6,
        "STOP_HUNT",
        "CLEAN"
    )

    # -------------------------
    # EVENT RISK (NON-GATING)
    # -------------------------
    velocity_spike = df["velocity"].abs() > df["velocity"].rolling(10).mean() * 2.5
    structure_break = df["structure_state"] == "RANGE"

    df["event_risk"] = velocity_spike & structure_break

    # -------------------------
    # VOLATILITY COMPRESSION
    # -------------------------
    long_vol = tr.rolling(100).mean()
    df["volatility_compression"] = df["atr_14"] / (long_vol + 1e-6)

    return df.fillna(0)
