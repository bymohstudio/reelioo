# core/quant/ml_bridge.py

import os
import logging
import pandas as pd
import xgboost as xgb

# FIXED IMPORT (correct module + correct function)
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class MLBridge:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LONG_PATH = os.path.join(BASE_DIR, "ml_models", "long_model.json")
    SHORT_PATH = os.path.join(BASE_DIR, "ml_models", "short_model.json")

    _LONG_MODEL = None
    _SHORT_MODEL = None

    @classmethod
    def _load_models(cls):
        if cls._LONG_MODEL and cls._SHORT_MODEL:
            return cls._LONG_MODEL, cls._SHORT_MODEL

        try:
            if os.path.exists(cls.LONG_PATH):
                cls._LONG_MODEL = xgb.Booster()
                cls._LONG_MODEL.load_model(cls.LONG_PATH)

            if os.path.exists(cls.SHORT_PATH):
                cls._SHORT_MODEL = xgb.Booster()
                cls._SHORT_MODEL.load_model(cls.SHORT_PATH)

            return cls._LONG_MODEL, cls._SHORT_MODEL

        except Exception as e:
            log.error(f"ML Model Load Failed: {e}")
            return None, None

    @classmethod
    def get_signal(cls, df: pd.DataFrame) -> dict:

        if df.empty:
            return {"bias": "NEUTRAL", "score": 50}

        # Use correct feature engineering
        df = generate_features(df)

        m_long, m_short = cls._load_models()
        if not m_long or not m_short:
            return {"bias": "NEUTRAL", "score": 50}

        try:
            last_row = df.iloc[[-1]][FEATURES].astype(float)
            dmat = xgb.DMatrix(last_row)

            p_long = float(m_long.predict(dmat)[0]) * 100
            p_short = float(m_short.predict(dmat)[0]) * 100

            CONF = 60  # relaxed threshold

            if p_long > CONF and p_long > p_short:
                return {"bias": "LONG", "score": round(p_long, 2)}

            if p_short > CONF and p_short > p_long:
                return {"bias": "SHORT", "score": round(p_short, 2)}

            # return real score instead of 50
            return {"bias": "NEUTRAL", "score": round(max(p_long, p_short), 2)}

        except Exception as e:
            log.error(f"Bridge Prediction Error: {e}")
            return {"bias": "NEUTRAL", "score": 50}
