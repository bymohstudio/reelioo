# core/quant/ml_bridge.py

import os
import logging
import pandas as pd
import xgboost as xgb
from .indicators import ML_FEATURES

log = logging.getLogger(__name__)


class MLBridge:
    _MODEL = None
    MODEL_PATH = os.path.join(os.getcwd(), "core", "quant", "ml_models", "crypto_edge.json")

    @classmethod
    def _load_model(cls):
        if cls._MODEL: return cls._MODEL
        if not os.path.exists(cls.MODEL_PATH): return None

        try:
            model = xgb.Booster()
            model.load_model(cls.MODEL_PATH)
            cls._MODEL = model
            return model
        except Exception as e:
            log.error(f"ML Model Load Failed: {e}")
            return None

    @classmethod
    def predict_proba(cls, df: pd.DataFrame) -> float:
        """
        Returns probability (0-100) that price will RISE.
        """
        model = cls._load_model()
        if not model or df.empty: return 50.0

        try:
            # Take the last row (latest candle)
            last_row = df.iloc[[-1]][ML_FEATURES]
            dmat = xgb.DMatrix(last_row)

            # Predict
            prob = model.predict(dmat)[0]  # Returns 0.0 to 1.0
            return float(prob * 100)
        except Exception as e:
            log.error(f"Prediction Error: {e}")
            return 50.0