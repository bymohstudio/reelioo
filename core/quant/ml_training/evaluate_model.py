# core/quant/evaluate_model.py
import sys

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import requests
import time
import os
from datetime import datetime, timedelta

# Import the NEW Asymmetric Logic
from core.quant.ml_training.feature_engineering import generate_features, generate_targets, FEATURES

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1h"
LOOKBACK = 90  # Last 3 months

current_dir = os.path.dirname(os.path.abspath(__file__))  # core/quant/ml_training
parent_dir = os.path.dirname(current_dir)                 # core/quant
MODEL_DIR = os.path.join(parent_dir, "ml_models")         # core/quant/ml_models

# 2. Fix Import Paths so we can run as a module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

PATHS = {
    "xgb_long": os.path.join(MODEL_DIR, "xgb_long.json"),
    "xgb_short": os.path.join(MODEL_DIR, "xgb_short.json"),
    "lgb_long": os.path.join(MODEL_DIR, "lgb_long.txt"),
    "lgb_short": os.path.join(MODEL_DIR, "lgb_short.txt"),
    "cat_long": os.path.join(MODEL_DIR, "cat_long.cbm"),
    "cat_short": os.path.join(MODEL_DIR, "cat_short.cbm"),
}


def fetch_data(symbol):
    print(f"   Fetching {symbol}...")
    start = int((datetime.now() - timedelta(days=LOOKBACK)).timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/klines"
    data = []
    try:
        current = start
        while True:
            params = {"symbol": symbol, "interval": INTERVAL, "startTime": current, "limit": 1000}
            res = requests.get(url, params=params, timeout=5).json()
            if not isinstance(res, list) or len(res) == 0: break
            data.extend(res)
            current = res[-1][6] + 1
            if len(data) >= 1500: break
            time.sleep(0.05)
    except:
        return pd.DataFrame()

    if not data: return pd.DataFrame()
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df


def load_models():
    models = {}
    try:
        if os.path.exists(PATHS['xgb_long']):
            models['xgb_long'] = xgb.Booster(model_file=PATHS['xgb_long'])
            models['xgb_short'] = xgb.Booster(model_file=PATHS['xgb_short'])
        if os.path.exists(PATHS['lgb_long']):
            models['lgb_long'] = lgb.Booster(model_file=PATHS['lgb_long'])
            models['lgb_short'] = lgb.Booster(model_file=PATHS['lgb_short'])
        if os.path.exists(PATHS['cat_long']):
            models['cat_long'] = CatBoostClassifier()
            models['cat_long'].load_model(PATHS['cat_long'])
            models['cat_short'] = CatBoostClassifier()
            models['cat_short'].load_model(PATHS['cat_short'])
        return models
    except:
        return None


def evaluate():
    print(f"\n🔍 EVALUATING LIVE PERFORMANCE (Threshold: 70%)")
    print(f"=================================================")

    models = load_models()
    if not models:
        print("❌ Models not found.")
        return

    # METRICS
    l_trades, l_wins = 0, 0
    s_trades, s_wins = 0, 0

    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df.empty: continue

        # Feature Engineering + New Targets
        df = generate_features(df)
        df = generate_targets(df)  # Applies the new "Crash" logic for validation

        X = df[FEATURES]
        dmat = xgb.DMatrix(X)

        # Predict
        try:
            p_l = (models['xgb_long'].predict(dmat) + models['lgb_long'].predict(X) + models['cat_long'].predict_proba(
                X)[:, 1]) / 3 * 100
            p_s = (models['xgb_short'].predict(dmat) + models['lgb_short'].predict(X) + models[
                                                                                            'cat_short'].predict_proba(
                X)[:, 1]) / 3 * 100
        except:
            continue

        # Simulate
        for i in range(len(df)):
            if pd.isna(df['target_long'].iloc[i]): continue

            score_l, score_s = p_l[i], p_s[i]

            # Long Logic (Sniper)
            if score_l > 70 and score_l > score_s:
                l_trades += 1
                if df['target_long'].iloc[i] == 1: l_wins += 1

            # Short Logic (Crash Hunter)
            elif score_s > 70 and score_s > score_l:
                s_trades += 1
                if df['target_short'].iloc[i] == 1: s_wins += 1

    print("\n📊 FINAL TEST RESULTS (Last 90 Days)")
    print("------------------------------------")
    print(f"🔹 LONG (Sniper):    {l_trades} trades | WR: {(l_wins / l_trades * 100) if l_trades else 0:.1f}%")
    print(f"🔸 SHORT (Crash):    {s_trades} trades | WR: {(s_wins / s_trades * 100) if s_trades else 0:.1f}%")

    if s_trades > 0 and (s_wins / s_trades) > 0.5:
        print("\n✅ SYSTEM IS GREEN. READY FOR DEPLOYMENT.")
    else:
        print("\n⚠️ SYSTEM NEEDS TUNING.")


if __name__ == "__main__":
    evaluate()