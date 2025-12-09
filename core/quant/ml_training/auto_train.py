# core/quant/ml_training/auto_train.py

import os
import sys
import xgboost as xgb
import pandas as pd
from dotenv import load_dotenv
import logging

# Setup path to run as script
sys.path.append(os.getcwd())
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("AutoTrain")

try:
    from core.quant.ml_training.fetch_data import DataFetcher
    from core.quant.ml_training.feature_engineering import FeatureEngineering
except ImportError:
    print("Run this script from the root project directory: python core/quant/ml_training/auto_train.py")
    sys.exit(1)

MODEL_DIR = os.path.join(os.getcwd(), "core", "quant", "ml_models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- EXPANDED INSTITUTIONAL UNIVERSE ---
UNIVERSE = {
    "EQUITY": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
        "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT", "TITAN"
    ],
    "FUTURES": ["NIFTY", "BANKNIFTY"],
    "OPTIONS": ["NIFTY", "BANKNIFTY"],
    "CRYPTO": ["BTC-INR", "ETH-INR"]
}


def train_market(market_type, symbols, days=1000):
    print(f"\n========================================")
    print(f"   TRAINING: {market_type}")
    print(f"========================================")

    if isinstance(symbols, str): symbols = [symbols]

    master_data = pd.DataFrame()

    # 1. Fetch & Aggregate Data
    for sym in symbols:
        try:
            # Decide fetch source logic
            fetch_market = "CRYPTO" if market_type == "CRYPTO" else "EQUITY"
            if market_type in ["FUTURES", "OPTIONS"]: fetch_market = "EQUITY"

            df = DataFetcher.fetch(sym, fetch_market, days)

            if df is not None and len(df) > 200:
                feats = FeatureEngineering.build(df)
                if not feats.empty:
                    master_data = pd.concat([master_data, feats])
                    print(f"  + {sym}: Added {len(feats)} rows")
            else:
                print(f"  - {sym}: No Data or Too Short")
        except Exception as e:
            print(f"  ! {sym}: Error ({e})")

    if master_data.empty:
        print(f"  FAILED: No valid training data for {market_type}")
        return

    # 2. Prepare Data
    print(f"  > Dataset Size: {len(master_data)} rows")
    X = master_data.drop(columns=["target"])
    y = master_data["target"]

    dall = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "eta": 0.05,
        "eval_metric": ["logloss", "error"],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 4
    }

    # 3. RUN CROSS-VALIDATION (The Scoring Step)
    print("  > Running 5-Fold Cross-Validation...")
    cv_results = xgb.cv(
        params,
        dall,
        num_boost_round=200,
        nfold=5,
        metrics=["logloss", "error"],
        early_stopping_rounds=20,
        verbose_eval=False,
        seed=42
    )

    # 4. Extract Scores
    best_logloss = cv_results['test-logloss-mean'].iloc[-1]
    best_error = cv_results['test-error-mean'].iloc[-1]
    accuracy = (1 - best_error) * 100
    optimal_rounds = cv_results.shape[0]

    print(f"  ----------------------------------------")
    print(f"  [SCORE] Est. Accuracy: {accuracy:.2f}%")
    print(f"  [SCORE] CV LogLoss:    {best_logloss:.4f}")
    print(f"  ----------------------------------------")

    # 5. Final Training (Production Model)
    # We train on 100% of data using the optimal rounds found by CV
    print(f"  > Finalizing model ({optimal_rounds} rounds)...")
    final_model = xgb.train(
        params,
        dall,
        num_boost_round=int(optimal_rounds * 1.1),  # Add 10% buffer
        verbose_eval=False
    )

    # 6. Save
    out_path = os.path.join(MODEL_DIR, f"{market_type.lower()}_edge.json")
    final_model.save_model(out_path)
    print(f"  [SAVED] {out_path}\n")


if __name__ == "__main__":
    print("\nStarting Institutional Training Pipeline...")

    for market, symbols in UNIVERSE.items():
        train_market(market, symbols)

    print("\nDONE. All models updated with CV scores.")