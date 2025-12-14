import os
import django
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT","MATICUSDT",
    "ARBUSDT","OPUSDT","INJUSDT","APTUSDT","ATOMUSDT",
    "NEARUSDT","FILUSDT","RNDRUSDT","SUIUSDT","SEIUSDT"]
INTERVAL = "1h"  # Changed to 1H for Swing Trading stability
LOOKBACK_DAYS = 720  # 24 Months of data
TARGET_PROFIT = 0.015  # 1.5% Move
TARGET_CANDLES = 6  # Within 6 Hours

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "core", "quant", "ml_models")
MODEL_PATH = os.path.join(MODEL_DIR, "crypto_edge.json")
META_PATH = os.path.join(MODEL_DIR, "edge_meta.json")

# Features List (MUST MATCH crypto_engine.py EXACTLY)
TITANIUM_FEATURES = [
    'rsi', 'squeeze', 'vol_z', 'trend_strength', 'atr_ratio',
    'bb_width', 'body_size'
]


# ---------------------------------------------------------
# 1. DEEP DATA LOADER
# ---------------------------------------------------------
class DeepDataLoader:
    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

    @staticmethod
    def fetch(symbol):
        print(f"   ⬇️  Fetching {LOOKBACK_DAYS} days for {symbol}...")
        start_ts = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
        end_ts = int(time.time() * 1000)
        data = []
        curr = start_ts

        while curr < end_ts:
            try:
                params = {"symbol": symbol, "interval": INTERVAL, "startTime": curr, "limit": 1500}
                res = requests.get(DeepDataLoader.BASE_URL, params=params, timeout=5).json()
                if not res or not isinstance(res, list): break
                data.extend(res)
                curr = res[-1][6] + 1  # Next start time
                time.sleep(0.05)  # Rate limit nice
            except:
                break

        if not data: return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "trades", "taker_base", "taker_quote", "ignore"
        ])

        # Numeric Conversion
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        return df


# ---------------------------------------------------------
# 2. TITANIUM MATH ENGINE
# ---------------------------------------------------------
def calculate_features(df):
    """
    Exact replica of the internal logic in crypto_engine.py
    """
    data = df.copy()
    close = data['close']

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    data['squeeze'] = (upper - lower) / sma20
    data['bb_width'] = data['squeeze']  # Alias

    # Volume Z-Score
    vol_mean = data['volume'].rolling(20).mean()
    vol_std = data['volume'].rolling(20).std()
    data['vol_z'] = (data['volume'] - vol_mean) / vol_std

    # Trend Strength
    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    data['trend_strength'] = (ema9 - ema21) / close * 100

    # ATR Ratio
    tr = pd.concat([
        data['high'] - data['low'],
        (data['high'] - close.shift()).abs(),
        (data['low'] - close.shift()).abs()
    ], axis=1).max(axis=1)
    data['atr_ratio'] = tr.rolling(14).mean() / close

    # Body Size
    data['body_size'] = (close - data['open']).abs() / close

    # TARGET GENERATION (The 'Answer Key')
    # Did price go up TARGET_PROFIT % within TARGET_CANDLES?
    # Uses rolling max of future highs to find profit hits
    future_highs = data['high'].rolling(TARGET_CANDLES).max().shift(-TARGET_CANDLES)
    data['target'] = (future_highs > close * (1 + TARGET_PROFIT)).astype(int)

    return data.dropna()


# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
def train_system():
    print(f"\n🚀 STARTING SUPREME TRAINING RUN")
    print(f"🎯 Target: +{TARGET_PROFIT * 100}% in {TARGET_CANDLES} candles ({INTERVAL})")

    master_train = []
    master_test = []

    # A. Fetch & Process All Symbols
    for sym in SYMBOLS:
        df = DeepDataLoader.fetch(sym)
        if df.empty: continue

        df = calculate_features(df)

        # Split 80/20 (Chronological split to prevent data leakage)
        split = int(len(df) * 0.85)
        master_train.append(df.iloc[:split])
        master_test.append(df.iloc[split:])

    if not master_train:
        print("❌ CRITICAL: No data gathered.")
        return

    # B. Combine Datasets
    train_df = pd.concat(master_train)
    test_df = pd.concat(master_test)

    X_train = train_df[TITANIUM_FEATURES]
    y_train = train_df['target']

    X_test = test_df[TITANIUM_FEATURES]
    y_test = test_df['target']

    print(f"\n📊 DATASET STATS:")
    print(f"   - Training Rows: {len(X_train):,}")
    print(f"   - Testing Rows:  {len(X_test):,}")
    print(f"   - Win Rate (Base): {(y_train.mean() * 100):.1f}% (If you bought randomly)")

    # C. Handle Class Imbalance
    # Calculate scale_pos_weight so the model cares about rare wins
    neg, pos = np.bincount(y_train)
    scale_weight = neg / pos
    print(f"   - Weighting: 1 Win = {scale_weight:.1f} Losses")

    # D. Train XGBoost
    print("\n🏋️ TRAINING NEURAL NETWORK (XGBOOST)...")
    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # E. Evaluate Performance
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # Calculate simulated Profit Factor on Test Data
    wins = preds * y_test  # Predicted 1 and was 1
    losses = preds * (1 - y_test)  # Predicted 1 and was 0
    profit_factor = (sum(wins) * 2.0) / (sum(losses) * 1.0)  # Assuming 2:1 RR

    print("\n🏆 FINAL REPORT CARD:")
    print(f"   ---------------------------")
    print(f"   ✅ AUC Score:      {auc:.3f} (0.5=Random, 1.0=God)")
    print(f"   ✅ Precision:      {precision:.2f}")
    print(f"   ✅ Recall:         {recall:.2f}")
    print(f"   ✅ Est. Profit F:  {profit_factor:.2f}")

    # F. Save Artifacts
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # 1. Save Model
    model.save_model(MODEL_PATH)
    print(f"\n💾 Model Saved: {MODEL_PATH}")

    # 2. Save Metadata (The Brain Instructions)
    meta = {
        "features": TITANIUM_FEATURES,
        "best_threshold": 0.65,  # Conservative threshold for high confidence
        "metrics": {
            "auc": round(float(auc), 3),
            "win_rate": round(float(precision * 100), 1),
            "profit_factor": round(float(profit_factor), 2),
            "test_samples": len(X_test)
        },
        "trained_date": time.strftime("%Y-%m-%d %H:%M")
    }

    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"💾 Metadata Saved: {META_PATH}")

    # G. Feature Importance
    print("\n🔑 TOP DRIVERS:")
    imps = model.feature_importances_
    for i in np.argsort(imps)[::-1]:
        print(f"   - {TITANIUM_FEATURES[i]:<15}: {imps[i]:.4f}")


if __name__ == "__main__":
    train_system()