from __future__ import annotations
import os

import xgboost as xgb

from .fetch_data import DataFetcher
from .feature_engineering import FeatureEngineering


class TrainFuturesModel:
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "ml_models", "futures_edge.json"
    )

    @classmethod
    def run(cls, symbol: str = "NIFTY"):
        print(f"[ML] Fetching index-underlying data for FUTURES: {symbol}")
        df = DataFetcher.fetch(symbol, market="FUTURES", days=365)

        if df is None or df.empty:
            raise ValueError(f"No futures-underlying data for {symbol}")

        print("[ML] Building futures features + labels…")
        df = FeatureEngineering.build(df)

        X = df.drop(columns=["target"])
        y = df["target"]

        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        print("[ML] Training futures XGBoost model…")
        booster = xgb.train(params, dtrain, num_boost_round=180)

        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        booster.save_model(cls.MODEL_PATH)
        print(f"[OK] Futures Model saved → {cls.MODEL_PATH}")
