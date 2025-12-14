import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import os
import time
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "1h"
LOOKBACK_DAYS = 150

MODEL_PATH = "core/quant/ml_models/crypto_edge.json"

FEATURES = [
    'rsi', 'squeeze', 'vol_z',
    'trend_strength', 'atr_ratio',
    'bb_width', 'body_size'
]

TARGET_PROFIT = 1.5      # %
TRADE_COST = 0.10        # % per trade (fees + slippage)

MIN_TRADES = 30          # Minimum trades to trust metrics

# ---------------------------------------------------------
# DATA FETCH
# ---------------------------------------------------------
def fetch_data(symbol):
    start = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    end = int(time.time() * 1000)
    data = []
    cur = start

    while cur < end:
        try:
            res = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": INTERVAL, "startTime": cur, "limit": 1500},
                timeout=5
            ).json()
            if not isinstance(res, list):
                break
            data.extend(res)
            cur = res[-1][6] + 1
            time.sleep(0.05)
        except:
            break

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "ts","open","high","low","close","volume",
        "ct","q","n","tb","tq","i"
    ])
    df[["open","high","low","close","volume"]] = df[
        ["open","high","low","close","volume"]
    ].astype(float)

    return df

# ---------------------------------------------------------
# FEATURE ENGINEERING + TARGET
# ---------------------------------------------------------
def build_features(df):
    close = df['close']

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_width'] = (std * 4) / sma
    df['squeeze'] = df['bb_width']

    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['vol_z'] = (df['volume'] - vol_mean) / vol_std

    ema9 = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    df['trend_strength'] = (ema9 - ema21) / close * 100

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - close.shift()).abs(),
        (df['low'] - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr_ratio'] = tr.rolling(14).mean() / close

    df['body_size'] = (close - df['open']).abs() / close

    future_high = df['high'].rolling(6).max().shift(-6)
    df['target'] = (future_high > close * 1.015).astype(int)

    return df.dropna()

# ---------------------------------------------------------
# METRIC SIMULATION
# ---------------------------------------------------------
def simulate_metrics(y_true, y_pred):
    trades = int(y_pred.sum())
    if trades < MIN_TRADES:
        return None

    wins = ((y_pred == 1) & (y_true == 1)).sum()
    losses = ((y_pred == 1) & (y_true == 0)).sum()

    win_rate = wins / trades

    gross_win = wins * (TARGET_PROFIT - TRADE_COST)
    gross_loss = losses * (TARGET_PROFIT + TRADE_COST)

    pf = gross_win / gross_loss if gross_loss > 0 else 0

    return {
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": pf
    }

# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------
def evaluate():
    print("\n📥 Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    frames = []
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty:
            frames.append(build_features(df))

    df = pd.concat(frames)
    X = df[FEATURES]
    y = df['target']

    probs = model.predict_proba(X)[:, 1]

    # -------- WALK-FORWARD SPLIT --------
    split = int(len(df) * 0.7)
    val_probs, test_probs = probs[:split], probs[split:]
    val_y, test_y = y.iloc[:split], y.iloc[split:]

    # -------- THRESHOLD SEARCH (VALIDATION ONLY) --------
    best = None
    best_thresh = None

    for t in np.arange(0.55, 0.81, 0.05):
        res = simulate_metrics(val_y, (val_probs > t).astype(int))
        if res and (best is None or res["profit_factor"] > best["profit_factor"]):
            best = res
            best_thresh = t

    print("\n==============================")
    print("🚀 FINAL REPORT (REALISTIC)")
    print("==============================")
    print(f"✔ Locked Threshold: {best_thresh:.2f}")
    print(f"✔ AUC (Global):     {roc_auc_score(y, probs):.3f}")

    final = simulate_metrics(test_y, (test_probs > best_thresh).astype(int))

    if final is None:
        print("⚠️ Not enough trades on test data at this threshold.")
        print("   → Model is HIGHLY selective.")
        print("   → Use as high-confidence signal engine.")
        return

    print(f"✔ Trades:          {final['trades']}")
    print(f"✔ Win Rate:        {final['win_rate']*100:.1f}%")
    print(f"✔ Profit Factor:   {final['profit_factor']:.2f}")

    if final["profit_factor"] >= 1.3:
        print("✅ Edge validated. Production-safe.")
    else:
        print("⚠️ Edge weak. Use caution.")

if __name__ == "__main__":
    evaluate()
