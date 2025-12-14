import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import time
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# CONFIG (MATCH TRAINING)
# ---------------------------------------------------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "ADAUSDT","AVAXUSDT","DOGEUSDT","LINKUSDT"
]

INTERVAL = "1h"
LOOKBACK_DAYS = 180

MODEL_PATH = "../ml_models/crypto_edge.json"

FEATURES = [
    'rsi','squeeze','vol_z',
    'trend_strength','atr_ratio',
    'bb_width','body_size'
]

TARGET_PROFIT = 0.015      # +1.5%
TARGET_CANDLES = 6         # within 6 hours
THRESHOLD = 0.75           # HIGH CONFIDENCE SIGNALS
MIN_SIGNALS = 30

# ---------------------------------------------------------
# BINANCE SAFE FETCH
# ---------------------------------------------------------
def fetch_data(symbol):
    start = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    end = int(time.time() * 1000)

    data, cur = [], start

    while cur < end:
        try:
            res = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": INTERVAL,
                    "startTime": cur,
                    "limit": 1500
                },
                timeout=5
            ).json()

            if not isinstance(res, list) or len(res) == 0:
                break

            last = res[-1]
            if not isinstance(last, list) or len(last) < 7:
                break

            data.extend(res)
            cur = int(last[6]) + 1
            time.sleep(0.05)

        except:
            break

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).iloc[:, :12]
    df.columns = [
        "ts","open","high","low","close","volume",
        "ct","q","n","tb","tq","i"
    ]

    df[["open","high","low","close","volume"]] = df[
        ["open","high","low","close","volume"]
    ].astype(float)

    return df

# ---------------------------------------------------------
# FEATURES + TARGET (EXACT TRAINING LOGIC)
# ---------------------------------------------------------
def build_features(df):
    required = {"open","high","low","close","volume"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

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

    # 🎯 TARGET = opportunity existence
    future_high = df['high'].rolling(TARGET_CANDLES).max().shift(-TARGET_CANDLES)
    df['target'] = (future_high > close * (1 + TARGET_PROFIT)).astype(int)

    return df.dropna()

# ---------------------------------------------------------
# SIGNAL EVALUATION
# ---------------------------------------------------------
def evaluate():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    all_probs = []
    all_targets = []

    for sym in SYMBOLS:
        df = fetch_data(sym)
        df = build_features(df)

        if df.empty:
            print(f"⚠️ Skipping {sym}")
            continue

        X = df[FEATURES]
        y = df['target']

        probs = model.predict_proba(X)[:, 1]

        # only evaluate HIGH CONFIDENCE signals
        mask = probs >= THRESHOLD

        all_probs.extend(probs[mask])
        all_targets.extend(y[mask])

    if len(all_targets) < MIN_SIGNALS:
        print("❌ Not enough signals for reliable evaluation.")
        return

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    wins = all_targets.sum()
    losses = len(all_targets) - wins
    win_rate = wins / len(all_targets)

    # simple PF: reward = +1, loss = -1
    profit_factor = wins / losses if losses > 0 else np.inf

    print("\n==============================")
    print("📊 SIGNAL-BASED MODEL EVALUATION")
    print("==============================")
    print(f"Threshold:      {THRESHOLD}")
    print(f"Signals:        {len(all_targets)}")
    print(f"Win Rate:       {win_rate*100:.1f}%")
    print(f"Profit Factor:  {profit_factor:.2f}")

    if profit_factor >= 1.5:
        print("✅ Strong signal edge. Production-safe.")
    else:
        print("⚠️ Weak signal edge. Raise threshold.")

if __name__ == "__main__":
    evaluate()
