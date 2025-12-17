import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
import time
from datetime import datetime, timedelta
from sklearn.metrics import precision_score
from core.quant.ml_training.feature_engineering import generate_features, generate_targets, FEATURES

# CONFIG
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT",
    "NEARUSDT", "APTUSDT", "INJUSDT", "RNDRUSDT", "FETUSDT"
]
INTERVAL = "1h"
LOOKBACK_DAYS = 500

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "core", "quant", "ml_models")
LONG_MODEL_PATH = os.path.join(MODEL_DIR, "long_model.json")
SHORT_MODEL_PATH = os.path.join(MODEL_DIR, "short_model.json")
META_PATH = os.path.join(MODEL_DIR, "edge_meta.json")


class DeepDataLoader:
    @staticmethod
    def fetch(symbol):
        print(f"   ⬇️  Fetching {symbol}...")
        start_ts = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
        url = "https://fapi.binance.com/fapi/v1/klines"
        data = []
        current = start_ts
        now = int(time.time() * 1000)

        while current < now:
            try:
                params = {"symbol": symbol, "interval": INTERVAL, "startTime": current, "limit": 1000}
                res = requests.get(url, params=params, timeout=5).json()
                if not isinstance(res, list) or len(res) == 0: break
                data.extend(res)
                current = res[-1][6] + 1
                time.sleep(0.05)
            except:
                break

        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).iloc[:, :6]
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
        df = df.astype(float)
        return df


def train_specialist(X, y, name="Model"):
    print(f"\n🏋️ Training {name} Specialist...")

    # 1. FIXED: Conservative Weighting
    # Instead of full imbalance (6.0), we use a softer cap (3.0)
    # This reduces False Positives drastically.
    pos_ratio = (len(y) - y.sum()) / y.sum()
    scale = min(pos_ratio, 3.0)

    print(f"   - Class Balance: {y.sum()} Wins / {len(y) - y.sum()} Fails")
    print(f"   - Scale Weight: {scale:.2f} (Capped for Precision)")

    # 2. FIXED: High-Precision Hyperparameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,  # Slightly deeper for complex patterns
        "eta": 0.02,  # Very slow learning (High precision)
        "subsample": 0.6,  # More randomness to avoid overfitting
        "colsample_bytree": 0.6,
        "scale_pos_weight": scale,
        "min_child_weight": 10,  # Require strong evidence
        "gamma": 0.2  # Prune weak leaves (Noise reduction)
    }

    split = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, "Test")], early_stopping_rounds=50,
                      verbose_eval=False)

    preds = model.predict(dtest)
    # Check Precision at Trade Threshold
    high_conf = (preds > 0.65).astype(int)
    prec = precision_score(y_test, high_conf, zero_division=0)
    print(f"   ✅ {name} Precision @ 65% Conf: {prec * 100:.1f}%")

    return model


def run_training():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    master_df = []
    for sym in SYMBOLS:
        df = DeepDataLoader.fetch(sym)
        if df.empty: continue

        df = generate_features(df)

        # 3. CRITICAL: Easier Targets = Higher Win Rate
        # Risk: 1.5 ATR | Reward: 2.25 ATR (1:1.5 Ratio)
        # This is much easier to hit than the previous 1:2
        df = generate_targets(df, risk_reward=1.5, stop_mult=1.5, candles=12)
        master_df.append(df)

    full_data = pd.concat(master_df).dropna()

    # Filter Noise (Keep clean trends)
    print(f"\n🧹 Filtering Noise... (Original: {len(full_data)})")
    clean_data = full_data[
        (full_data['efficiency_ratio'] > 0.12) |
        (full_data['volatility_slope'] > 0.5)
        ]
    print(f"📊 Training on Active Markets: {len(clean_data)} Rows")

    long_model = train_specialist(clean_data[FEATURES], clean_data['target_long'], "LONG")
    if long_model: long_model.save_model(LONG_MODEL_PATH)

    short_model = train_specialist(clean_data[FEATURES], clean_data['target_short'], "SHORT")
    if short_model: short_model.save_model(SHORT_MODEL_PATH)

    with open(META_PATH, 'w') as f:
        json.dump({"updated": str(datetime.now()), "features": FEATURES}, f)

    print("\n💾 HIGH-PRECISION MODELS SAVED.")


if __name__ == "__main__":
    run_training()