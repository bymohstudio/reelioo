from __future__ import annotations
import os

import xgboost as xgb

from .fetch_data import DataFetcher
from .feature_engineering import FeatureEngineering


class TrainEquityModel:
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "ml_models", "equity_edge.json"
    )

    @classmethod
    def run(cls, symbol: str = "RELIANCE"):
        print(f"[ML] Fetching equity data for {symbol}")
        df = DataFetcher.fetch(symbol, market="EQUITY", days=365)

        if df is None or df.empty:
            raise ValueError(f"No equity data for {symbol}")

        print("[ML] Building equity features + labels…")
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

        print("[ML] Training equity XGBoost model…")
        booster = xgb.train(params, dtrain, num_boost_round=220)

        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        booster.save_model(cls.MODEL_PATH)
        print(f"[OK] Equity Model saved → {cls.MODEL_PATH}")
