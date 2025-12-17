# core/quant/indicators.py

# ------------------------------------------------------------
# COMPATIBILITY BRIDGE
# This file ensures that older modules (like BacktestEngine)
# use the NEW centralized math from feature_engineering.py
# ------------------------------------------------------------

import pandas as pd
from .ml_training.feature_engineering import compute_indicators, FEATURES

# Re-export FEATURES so other files can find them
ML_FEATURES = FEATURES

def compute_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper: Redirects to the new Feature Engineering engine.
    """
    return compute_indicators(df)