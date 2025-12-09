# core/quant/ml_bridge.py

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Import FEATURES to know what to align against
try:
    from .ml_training.feature_engineering import FEATURES
except ImportError:
    FEATURES = ["close_over_ma20", "rsi_14"]  # Fallback

log = logging.getLogger(__name__)

_LOADED_MODELS = {}


def _get_model_path(market: str):
    base = os.path.join(os.getcwd(), "core", "quant", "ml_models")
    return os.path.join(base, f"{market.lower()}_edge.json")


def _load_model(market: str):
    if market in _LOADED_MODELS: return _LOADED_MODELS[market]
    path = _get_model_path(market)
    if not os.path.exists(path): return None
    if xgb:
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            _LOADED_MODELS[market] = booster
            return booster
        except Exception as e:
            log.error(f"Model Load Error: {e}")
            return None
    return None


def _heuristic_predict(data) -> float:
    # Basic logic if ML fails
    score = 50.0
    try:
        # data could be dict or df
        if isinstance(data, pd.DataFrame):
            last = data.iloc[-1]
        else:
            last = data

        if "close" in last and "ma_50" in last:
            if last["close"] > last["ma_50"]:
                score += 10
            else:
                score -= 10
        if "rsi_14" in last:
            if last["rsi_14"] > 60: score -= 5
            if last["rsi_14"] < 40: score += 5
    except:
        pass
    return max(0.0, min(100.0, score)) / 100.0


def predict_directional_edge(market_type: str, symbol: str, trade_style: str, data) -> float:
    """
    Robust Prediction Bridge.
    Accepts data as DataFrame OR Dict.
    Aligns columns to match the trained model.
    """
    if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, dict) and not data):
        return 50.0

    # 1. Standardize to DataFrame row
    if isinstance(data, pd.DataFrame):
        # If full history passed (e.g. from backtest), take last row
        input_row = data.iloc[[-1]].copy()
    else:
        # If dict passed (e.g. from live engine)
        input_row = pd.DataFrame([data])

    # 2. Load Model
    model = _load_model(market_type)

    # 3. Align Features & Predict
    if model and xgb:
        try:
            # Create a DF with exactly the columns the model wants (FEATURES list)
            aligned_df = pd.DataFrame(index=input_row.index)

            # Fill columns
            for col in FEATURES:
                if col in input_row.columns:
                    aligned_df[col] = input_row[col]
                else:
                    aligned_df[col] = 0.0  # Missing feature = 0

            # Predict
            dmat = xgb.DMatrix(aligned_df)
            prob = float(model.predict(dmat)[0])
            return prob * 100.0

        except Exception as e:
            log.error(f"XGBoost Error: {e}")
            return _heuristic_predict(input_row) * 100.0
    else:
        return _heuristic_predict(input_row) * 100.0