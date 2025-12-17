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

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1h"
LOOKBACK = 180  # Evaluate last 6 months

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "core", "quant", "ml_models")

PATHS = {
    "xgb_long": os.path.join(MODEL_DIR, "xgb_long.json"),
    "xgb_short": os.path.join(MODEL_DIR, "xgb_short.json"),
    "lgb_long": os.path.join(MODEL_DIR, "lgb_long.txt"),
    "lgb_short": os.path.join(MODEL_DIR, "lgb_short.txt"),
    "cat_long": os.path.join(MODEL_DIR, "cat_long.cbm"),
    "cat_short": os.path.join(MODEL_DIR, "cat_short.cbm"),
}


def fetch_data(symbol):
    print(f"Fetching {symbol}...")
    start = int((datetime.now() - timedelta(days=LOOKBACK)).timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/klines"
    data = []
    try:
        # Fetch in chunks
        current = start
        while True:
            params = {"symbol": symbol, "interval": INTERVAL, "startTime": current, "limit": 1000}
            res = requests.get(url, params=params).json()
            if not isinstance(res, list) or len(res) == 0: break
            data.extend(res)
            current = res[-1][6] + 1
            if len(data) >= 2000: break  # Limit for speed
            time.sleep(0.1)
    except:
        pass

    if not data: return pd.DataFrame()

    # Process Data
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df


def load_models():
    models = {}
    try:
        # XGB
        models['xgb_long'] = xgb.Booster(model_file=PATHS['xgb_long'])
        models['xgb_short'] = xgb.Booster(model_file=PATHS['xgb_short'])
        # LGB
        models['lgb_long'] = lgb.Booster(model_file=PATHS['lgb_long'])
        models['lgb_short'] = lgb.Booster(model_file=PATHS['lgb_short'])
        # CAT
        models['cat_long'] = CatBoostClassifier()
        models['cat_long'].load_model(PATHS['cat_long'])
        models['cat_short'] = CatBoostClassifier()
        models['cat_short'].load_model(PATHS['cat_short'])
        return models
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None


def evaluate():
    print(f"\n🔍 EVALUATING ENSEMBLE SYSTEM (1.5 R:R)")

    models = load_models()
    if not models: return

    total_trades = 0
    wins = 0

    # Track stats per coin
    coin_stats = {}

    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df.empty: continue

        # Feature Engineering
        df = generate_features(df)

        # Targets (1.5 R:R to match training)
        df = generate_targets(df, risk_reward=1.5, stop_mult=1.5, candles=12)

        # Prepare Data Structures for fast prediction
        X = df[FEATURES]
        dmat = xgb.DMatrix(X)

        # --- GET PREDICTIONS ---
        # Longs
        p_xl = models['xgb_long'].predict(dmat)
        p_ll = models['lgb_long'].predict(X)
        p_cl = models['cat_long'].predict_proba(X)[:, 1]

        # Shorts
        p_xs = models['xgb_short'].predict(dmat)
        p_ls = models['lgb_short'].predict(X)
        p_cs = models['cat_short'].predict_proba(X)[:, 1]

        # Averaging (Ensemble Logic)
        ens_long = (p_xl + p_ll + p_cl) / 3.0 * 100
        ens_short = (p_xs + p_ls + p_cs) / 3.0 * 100

        sym_trades = 0
        sym_wins = 0

        # Simulate Trading Day by Day
        for i in range(len(df)):
            # 1. Volatility Filters (Matching 'DAY' mode logic)
            # Must match the "Relaxed" training logic
            eff = df['efficiency_ratio'].iloc[i]
            vol = df['volatility_slope'].iloc[i]

            # Filter: Efficiency > 0.08 OR Volatility > 0.3 (Allows small moves)
            if eff < 0.08 and vol < 0.3: continue

            p_l = ens_long[i]
            p_s = ens_short[i]

            # Threshold: 65% (Standard Day Trading Mode)
            # Long Signal
            if p_l > 65 and p_l > (p_s + 5):
                total_trades += 1
                sym_trades += 1
                if df['target_long'].iloc[i] == 1:
                    wins += 1
                    sym_wins += 1

            # Short Signal
            elif p_s > 65 and p_s > (p_l + 5):
                total_trades += 1
                sym_trades += 1
                if df['target_short'].iloc[i] == 1:
                    wins += 1
                    sym_wins += 1

        # Log individual coin stats
        wr = (sym_wins / sym_trades * 100) if sym_trades > 0 else 0
        coin_stats[sym] = f"{sym_trades} trades, {wr:.1f}% WR"

    # 3. REPORT RESULTS
    print("\n📊 ASSET BREAKDOWN:")
    for sym, stat in coin_stats.items():
        print(f"   - {sym}: {stat}")

    if total_trades == 0:
        print("\n⚠️ No signals found. Try lowering threshold to 60% (Scalp Mode).")
    else:
        wr = (wins / total_trades) * 100
        print("\n🏆 FINAL ENSEMBLE RESULTS")
        print(f"   Total Signals: {total_trades}")
        print(f"   Overall Win Rate: {wr:.1f}%")

        # Profitability Check (Breakeven for 1.5 R:R is 40%)
        if wr > 40:
            print(f"✅ PROFITABLE SYSTEM (Edge: +{wr - 40:.1f}%)")
        else:
            print("⚠️ System Breakeven/Loss. Needs higher precision.")


if __name__ == "__main__":
    evaluate()