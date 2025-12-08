import os
import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

MODEL_MAP = {
    "EQUITY": "equity_edge.json",
    "FUTURES": "futures_edge.json",
    "OPTIONS": "options_edge.json",
    "CRYPTO": "crypto_edge.json",
}

_loaded_models: Dict[str, xgb.Booster] = {}


def _load_model_safe(market_type: str) -> Optional[xgb.Booster]:
    market_type = (market_type or "").upper()
    if market_type in _loaded_models:
        return _loaded_models[market_type]

    filename = MODEL_MAP.get(market_type)
    if not filename:
        return None

    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"[ML WARNING] Model file missing: {filename}")
        return None

    try:
        booster = xgb.Booster()
        booster.load_model(path)
        _loaded_models[market_type] = booster
        return booster
    except Exception as e:
        print(f"[ML ERROR] Could not load {filename}: {e}")
        return None


def _align_features_safe(feature_row: Dict[str, Any], booster: xgb.Booster) -> xgb.DMatrix:
    """
    AUTO-FIXER: Aligns input data to match Model requirements exactly.
    """
    # 1. Ask the model what features it needs
    expected_features = booster.feature_names

    # Fallback if model has no saved feature names
    if not expected_features:
        data = np.array([list(feature_row.values())], dtype=np.float32)
        return xgb.DMatrix(data, feature_names=list(feature_row.keys()))

    # 2. Build the exact vector the model wants
    aligned_vector = []
    missing_count = 0

    for name in expected_features:
        # If we have the feature, use it. If not, fill with 0.0
        if name in feature_row:
            val = float(feature_row[name])
        else:
            val = 0.0
            missing_count += 1

        # Handle NaN/Inf safety
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        aligned_vector.append(val)

    # 3. Log if we had to fix things (Debugging only)
    if missing_count > 0:
        print(f"[ML REPAIR] Auto-filled {missing_count} missing features to prevent crash.")

    # 4. Return correct DMatrix
    data = np.array([aligned_vector], dtype=np.float32)
    return xgb.DMatrix(data, feature_names=expected_features)


def predict_directional_edge(market_type: str, *args, **kwargs) -> float:
    """
    Main Prediction Function
    """
    # Extract dict from args/kwargs
    feature_row = None
    for arg in args:
        if isinstance(arg, dict): feature_row = arg
    if not feature_row:
        for k in ["features", "feature_row", "snapshot"]:
            if isinstance(kwargs.get(k), dict): feature_row = kwargs[k]

    if not feature_row:
        return 50.0

    booster = _load_model_safe(market_type)
    if booster is None:
        return 50.0

    try:
        # --- THE FIX IS HERE ---
        # We use the safe aligner instead of raw DMatrix
        dmat = _align_features_safe(feature_row, booster)

        prob = float(booster.predict(dmat)[0])
        return round(prob * 100.0, 2)
    except Exception as e:
        print(f"[ML CRASH PREVENTED] {e}")
        return 50.0