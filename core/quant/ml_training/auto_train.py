import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "ml_models")
META_PATH = os.path.join(MODEL_DIR, "edge_meta.json")

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)


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
        # Ensure we get Taker Buy Volume (Index 9)
        df = pd.DataFrame(data).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
        df.columns = ["ts", "open", "high", "low", "close", "volume", "taker_base"]
        df = df.astype(float)
        return df


def train_xgboost(X_train, y_train, X_test, y_test, name, scale_pos_weight):
    print(f"   🚀 Training XGBoost ({name})...")
    params = {
        "objective": "binary:logistic", "eval_metric": "auc", "max_depth": 6,
        "eta": 0.02, "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight, "min_child_weight": 3, "nthread": 4
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtest, "Test")],
                      early_stopping_rounds=50, verbose_eval=False)
    model.save_model(os.path.join(MODEL_DIR, f"xgb_{name.lower()}.json"))
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, name, scale_pos_weight):
    print(f"   🍃 Training LightGBM ({name})...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.03, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 5, "scale_pos_weight": scale_pos_weight,
        "verbose": -1, "nthread": 4
    }
    model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dtest],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    model.save_model(os.path.join(MODEL_DIR, f"lgb_{name.lower()}.txt"))
    return model


def train_catboost(X_train, y_train, X_test, y_test, name, scale_pos_weight):
    print(f"   🐱 Training CatBoost ({name})...")
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6, scale_pos_weight=scale_pos_weight,
        eval_metric='AUC', verbose=0, allow_writing_files=False, thread_count=4
    )
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
    model.save_model(os.path.join(MODEL_DIR, f"cat_{name.lower()}.cbm"))
    return model


def evaluate_ensemble(xgb_m, lgb_m, cat_m, X_test, y_test, threshold=0.60):
    p_xgb = xgb_m.predict(xgb.DMatrix(X_test))
    p_lgb = lgb_m.predict(X_test)
    p_cat = cat_m.predict_proba(X_test)[:, 1]
    avg_prob = (p_xgb + p_lgb + p_cat) / 3.0
    preds = (avg_prob > threshold).astype(int)
    return precision_score(y_test, preds, zero_division=0)


def run_training():
    master_df = []
    print("⏳ Loading Data...")
    for sym in SYMBOLS:
        df = DeepDataLoader.fetch(sym)
        if df.empty: continue
        df = generate_features(df)

        # TARGETS: Now detecting VOLATILITY EXPLOSIONS (Regime)
        df = generate_targets(df, risk_reward=2.0, stop_mult=1.5, candles=12)
        master_df.append(df)

    full_data = pd.concat(master_df).dropna()

    # We filter for at least mild volatility to train the model on active markets
    print(f"\n🧹 Filtering for Active Markets... (Original: {len(full_data)})")

    clean_data = full_data[
        (full_data['efficiency_ratio'] > 0.05) |
        (full_data['volatility_slope'] > 0.1)
        ]

    print(f"📊 High-Quality Training Set: {len(clean_data)} Rows")

    split = int(len(clean_data) * 0.85)
    X = clean_data[FEATURES]

    # Train Upside Volatility (Long)
    y_long = clean_data['target_long']
    scale_long = min((len(y_long) - y_long.sum()) / (y_long.sum() + 1), 5.0)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_l, y_test_l = y_long.iloc[:split], y_long.iloc[split:]

    print("\n🔹 TRAINING LONG REGIME ENSEMBLE")
    xgb_l = train_xgboost(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    lgb_l = train_lightgbm(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    cat_l = train_catboost(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    print(f"✅ LONG Regime Precision: {evaluate_ensemble(xgb_l, lgb_l, cat_l, X_test, y_test_l) * 100:.1f}%")

    # Train Downside Volatility (Short)
    y_short = clean_data['target_short']
    scale_short = min((len(y_short) - y_short.sum()) / (y_short.sum() + 1), 5.0)
    y_train_s, y_test_s = y_short.iloc[:split], y_short.iloc[split:]

    print("\n🔸 TRAINING SHORT REGIME ENSEMBLE")
    xgb_s = train_xgboost(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    lgb_s = train_lightgbm(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    cat_s = train_catboost(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    print(f"✅ SHORT Regime Precision: {evaluate_ensemble(xgb_s, lgb_s, cat_s, X_test, y_test_s) * 100:.1f}%")

    with open(META_PATH, 'w') as f:
        json.dump({"updated": str(datetime.now()), "features": FEATURES}, f)
    print("\n💾 ALL MODELS SAVED.")


if __name__ == "__main__":
    run_training()