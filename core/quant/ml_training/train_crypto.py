# core/quant/ml_training/train_crypto.py

import os
import xgboost as xgb
import logging
from .fetch_data import DataFetcher
from .feature_engineering import FeatureEngineering, FEATURES

log = logging.getLogger("TrainCrypto")


class TrainCryptoModel:
    # Save as 'crypto_edge.json' to be loaded by the engine
    MODEL_PATH = os.path.join(os.getcwd(), "core", "quant", "ml_models", "crypto_edge.json")

    @classmethod
    def run(cls, symbol: str = "BTC"):
        log.info(f"--- Training Crypto Model on {symbol} ---")

        # 1. Get Data
        df = DataFetcher.fetch(symbol, interval="1h")
        if df.empty:
            log.error("No data found.")
            return

        # 2. Build Features
        df = FeatureEngineering.build(df)
        log.info(f"Training on {len(df)} samples...")

        X = df[FEATURES]
        y = df["target"]

        # 3. Train XGBoost
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 0.8
        }

        model = xgb.train(params, dtrain, num_boost_round=100)

        # 4. Save
        # Ensure dir exists
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
        model.save_model(cls.MODEL_PATH)
        log.info(f"✅ Model saved to {cls.MODEL_PATH}")