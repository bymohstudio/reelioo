from __future__ import annotations
import os

import xgboost as xgb

from .fetch_data import DataFetcher
from .feature_engineering import FeatureEngineering


class TrainOptionsModel:
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "ml_models", "options_edge.json"
    )

    @classmethod
    def run(cls, underlying: str = "NIFTY"):
        print(f"[ML] Training OPTIONS model on underlying index: {underlying}")
        df = DataFetcher.fetch(underlying, market="OPTIONS", days=365)

        if df is None or df.empty:
            raise ValueError(f"No options-underlying data for {underlying}")

        print("[ML] Building options (synthetic) features + labels…")
        df = FeatureEngineering.build(df)

        X = df.drop(columns=["target"])
        y = df["target"]

        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        print("[ML] Training options XGBoost model…")
        booster = xgb.train(params, dtrain, num_boost_round=160)

        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        booster.save_model(cls.MODEL_PATH)
        print(f"[OK] Options Model saved → {cls.MODEL_PATH}")
