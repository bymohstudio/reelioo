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

from core.quant.ml_training.feature_engineering import generate_features, generate_targets, FEATURES

# --- CONFIGURATION (MATCHING CRYPTO_ENGINE) ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1h"
LOOKBACK = 90
THRESHOLD = 75.0  # <--- UPDATED TO MATCH ENGINE (Was 70.0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(parent_dir, "ml_models")

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
    print(f"\n🔍 EVALUATING PURIST PERFORMANCE (Threshold: {THRESHOLD}%)")
    print(f"=================================================")

    models = load_models()
    if not models:
        print("❌ Models not found.")
        return

    l_trades, l_wins = 0, 0
    s_trades, s_wins = 0, 0

    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df.empty: continue

        df = generate_features(df)
        df = generate_targets(df)

        X = df[FEATURES]
        dmat = xgb.DMatrix(X)

        try:
            p_l = (models['xgb_long'].predict(dmat) + models['lgb_long'].predict(X) + models['cat_long'].predict_proba(X)[:, 1]) / 3 * 100
            p_s = (models['xgb_short'].predict(dmat) + models['lgb_short'].predict(X) + models['cat_short'].predict_proba(X)[:, 1]) / 3 * 100
        except:
            continue

        for i in range(len(df)):
            if pd.isna(df['target_long'].iloc[i]): continue

            score_l, score_s = p_l[i], p_s[i]

            # --- PURIST LOGIC (MATCHING ENGINE) ---
            # Threshold raised to 75.0
            # No +5 artificial boosts added here

            # Long Logic
            if score_l > THRESHOLD and score_l > score_s:
                l_trades += 1
                if df['target_long'].iloc[i] == 1: l_wins += 1

            # Short Logic
            elif score_s > THRESHOLD and score_s > score_l:
                s_trades += 1
                if df['target_short'].iloc[i] == 1: s_wins += 1

    print(f"\n📊 FINAL PURIST RESULTS (Last 90 Days - >{THRESHOLD}%)")
    print("------------------------------------")
    print(f"🔹 LONG (Sniper):    {l_trades} trades | WR: {(l_wins / l_trades * 100) if l_trades else 0:.1f}%")
    print(f"🔸 SHORT (Crash):    {s_trades} trades | WR: {(s_wins / s_trades * 100) if s_trades else 0:.1f}%")

    if s_trades > 0 and (s_wins / s_trades) > 0.5:
        print("\n✅ SYSTEM ALIGNED. THIS REFLECTS REALITY.")
    else:
        print("\n⚠️ SYSTEM NEEDS TUNING.")

if __name__ == "__main__":
    evaluate()