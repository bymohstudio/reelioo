import os
import sys
import django
import json
import warnings

# --- 1. SETUP DJANGO ENVIRONMENT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reelioo.settings')
django.setup()

# --- 2. IMPORTS ---
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, roc_auc_score
from datetime import datetime

from core.quant.ml_training.feature_engineering import generate_features, generate_targets, FEATURES
from core.quant.ml_training.training_data_loader import TrainingDataLoader

warnings.filterwarnings("ignore")

# CONFIG
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "MATICUSDT",
    "NEARUSDT", "APTUSDT", "INJUSDT", "RNDRUSDT", "FETUSDT"
]
INTERVAL = "1h"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "ml_models")
META_PATH = os.path.join(MODEL_DIR, "edge_meta.json")

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)


class DeepDataLoader:
    @staticmethod
    def fetch(symbol):
        df = TrainingDataLoader.fetch_deep_history(symbol, limit=50000)
        if df.empty: return None
        return df


def train_xgboost(X_train, y_train, X_test, y_test, label, scale_pos_weight):
    print(f"   🚀 XGBoost ({label}) on RTX 4060...")

    # Optimized for Asymmetric Targets
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 10,
        'learning_rate': 0.015 if label == "LONG" else 0.02,  # Shorts learn faster
        'n_estimators': 4000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'device': 'cuda',  # GPU
        'gamma': 0.5,
        'alpha': 0.5,
        'lambda': 1.0,
        'early_stopping_rounds': 150
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model.save_model(os.path.join(MODEL_DIR, f"xgb_{label.lower()}.json"))
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, label, scale_pos_weight):
    print(f"   🍃 LightGBM ({label}) - CPU Optimized...")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 80,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_child_samples': 50,
        'verbose': -1,
        'n_estimators': 3000
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(0)]
    )
    model.booster_.save_model(os.path.join(MODEL_DIR, f"lgb_{label.lower()}.txt"))
    return model


def train_catboost(X_train, y_train, X_test, y_test, label, scale_pos_weight):
    print(f"   🐱 CatBoost ({label}) on RTX 4060...")

    model = CatBoostClassifier(
        iterations=4000,
        depth=10,
        learning_rate=0.02,
        loss_function='Logloss',
        eval_metric='AUC',
        scale_pos_weight=scale_pos_weight,
        task_type="GPU",
        devices='0',
        verbose=0,
        early_stopping_rounds=150,
        l2_leaf_reg=5,
        border_count=128
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    model.save_model(os.path.join(MODEL_DIR, f"cat_{label.lower()}.cbm"))
    return model


def evaluate_ensemble(xgb_m, lgb_m, cat_m, X_test, y_test):
    p1 = xgb_m.predict_proba(X_test)[:, 1]
    p2 = lgb_m.predict_proba(X_test)[:, 1]
    p3 = cat_m.predict_proba(X_test)[:, 1]

    final_prob = (p1 + p2 + p3) / 3

    print("\n   🎯 SNIPER SCOPE CALIBRATION (Normalized):")
    print(f"      Avg Prob: {np.mean(final_prob) * 100:.1f}% | Max Prob: {np.max(final_prob) * 100:.1f}%")

    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    for thresh in thresholds:
        preds = (final_prob > thresh).astype(int)
        count = np.sum(preds)
        if count > 0:
            prec = precision_score(y_test, preds, zero_division=0)
            print(f"      > {int(thresh * 100)}% Conf: Win Rate {prec * 100:.1f}%  ({count} trades)")
        else:
            print(f"      > {int(thresh * 100)}% Conf: --  (0 trades)")

    return precision_score(y_test, (final_prob > 0.65).astype(int), zero_division=0)


def save_metadata():
    meta_data = {
        "updated": str(datetime.now()),
        "features": FEATURES,
        "note": "RTX 4060 Asymmetric Targets"
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta_data, f)
    print(f"📝 Metadata updated at: {META_PATH}")


def run_training():
    print("🧹 Filtering for Active Markets...")
    all_data = []

    for sym in SYMBOLS:
        df = DeepDataLoader.fetch(sym)
        if df is None: continue

        try:
            df = generate_features(df)
            # Use Asymmetric Logic (Hardcoded in function now)
            df = generate_targets(df)
            df['symbol'] = sym
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    if not all_data:
        print("❌ No Data Found.")
        return

    full_data = pd.concat(all_data)
    full_data.replace([np.inf, -np.inf], 0, inplace=True)
    full_data.dropna(inplace=True)

    if 'volatility_slope' in full_data.columns:
        full_data = full_data[full_data['volatility_slope'] > 0.05]

    print(f"📊 GPU Training Set: {len(full_data)} Rows")

    X = full_data[FEATURES]
    split = int(len(X) * 0.85)

    # --- LONG TRAINING ---
    y_long = full_data['target_long']
    pos_count = y_long.sum()
    scale_long = np.sqrt((len(y_long) - pos_count) / (pos_count + 1)) * 1.5

    print(f"\n⚖️  Class Balance LONG: Scaled to {scale_long:.2f}")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_l, y_test_l = y_long.iloc[:split], y_long.iloc[split:]

    print("\n🔹 TRAINING LONG REGIME ENSEMBLE (GPU ACCELERATED)")
    xgb_l = train_xgboost(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    lgb_l = train_lightgbm(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    cat_l = train_catboost(X_train, y_train_l, X_test, y_test_l, "LONG", scale_long)
    evaluate_ensemble(xgb_l, lgb_l, cat_l, X_test, y_test_l)

    # --- SHORT TRAINING ---
    y_short = full_data['target_short']
    pos_count = y_short.sum()
    scale_short = np.sqrt((len(y_short) - pos_count) / (pos_count + 1)) * 1.5

    print(f"\n⚖️  Class Balance SHORT: Scaled to {scale_short:.2f}")

    y_train_s, y_test_s = y_short.iloc[:split], y_short.iloc[split:]

    print("\n🔸 TRAINING SHORT REGIME ENSEMBLE (GPU ACCELERATED)")
    xgb_s = train_xgboost(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    lgb_s = train_lightgbm(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    cat_s = train_catboost(X_train, y_train_s, X_test, y_test_s, "SHORT", scale_short)
    evaluate_ensemble(xgb_s, lgb_s, cat_s, X_test, y_test_s)

    save_metadata()
    print("\n💾 ALL MODELS SAVED.")


if __name__ == "__main__":
    run_training()