import pandas as pd
import numpy as np
import xgboost as xgb
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
LOOKBACK = 180

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LONG_PATH = os.path.join(BASE_DIR, "core", "quant", "ml_models", "long_model.json")
SHORT_PATH = os.path.join(BASE_DIR, "core", "quant", "ml_models", "short_model.json")


def fetch_data(symbol):
    print(f"Fetching {symbol}...")
    start = int((datetime.now() - timedelta(days=LOOKBACK)).timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/klines"
    data = []
    try:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": start, "limit": 1500}
        res = requests.get(url, params=params).json()
        if isinstance(res, list): data.extend(res)
    except:
        pass

    if not data: return pd.DataFrame()

    # Process Data
    df = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df


def evaluate():
    print(f"\n🔍 EVALUATING HIGH PRECISION LOGIC (1.5 R:R)")

    # 1. LOAD MODELS (The missing part fixed here)
    if not os.path.exists(LONG_PATH) or not os.path.exists(SHORT_PATH):
        print("❌ Models not found. Please run auto_train.py first.")
        return

    try:
        m_long = xgb.Booster()
        m_long.load_model(LONG_PATH)

        m_short = xgb.Booster()
        m_short.load_model(SHORT_PATH)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    total_trades = 0
    wins = 0

    # 2. RUN SIMULATION
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df.empty: continue

        # Feature Engineering
        df = generate_features(df)

        # Generate Targets (1.5 Risk Reward to match training)
        df = generate_targets(df, risk_reward=1.5, stop_mult=1.5, candles=12)

        # Predict
        dmat = xgb.DMatrix(df[FEATURES])
        preds_long = m_long.predict(dmat) * 100
        preds_short = m_short.predict(dmat) * 100

        # Iterate Row by Row
        for i in range(len(df)):
            # Filter matches Engine Logic
            eff = df['efficiency_ratio'].iloc[i]
            vol = df['volatility_slope'].iloc[i]

            # Skip choppy markets
            if eff < 0.15 and vol < 0.1: continue

            p_l = preds_long[i]
            p_s = preds_short[i]

            # Threshold matches Engine (65%)
            if p_l > 65 and p_l > (p_s + 10):
                total_trades += 1
                if df['target_long'].iloc[i] == 1: wins += 1

            elif p_s > 65 and p_s > (p_l + 10):
                total_trades += 1
                if df['target_short'].iloc[i] == 1: wins += 1

    # 3. REPORT RESULTS
    if total_trades == 0:
        print("⚠️ No signals found. Market might be too choppy for current filters.")
    else:
        wr = (wins / total_trades) * 100
        print("\n🏆 RESULTS")
        print(f"   Signals:   {total_trades}")
        print(f"   Win Rate:  {wr:.1f}%")

        if wr > 50:
            print("✅ HIGHLY PROFITABLE (With 1.5 RR, >40% is profit)")
        else:
            print("⚠️ Break-even or Loss. Adjust thresholds.")


if __name__ == "__main__":
    evaluate()