# core/quant/ml_training/auto_train.py

from __future__ import annotations

import os
import json
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from .fetch_data import DataFetcher
from .feature_engineering import FeatureEngineering

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# ---------------------------------------------------------------------
# Helper: market-specific config sets
# ---------------------------------------------------------------------


def _get_market_configs(market_type: str) -> List[Dict[str, Any]]:
    """
    Returns a list of hyperparameter configs tuned per market.
    """

    mt = (market_type or "").upper()

    # Default generic configs (safe fallback)
    generic = [
        {
            "desc": "Conservative",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3.0,
            "reg_lambda": 2.0,
            "gamma": 0.1,
        },
        {
            "desc": "Balanced",
            "max_depth": 6,
            "eta": 0.10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2.0,
            "reg_lambda": 1.0,
            "gamma": 0.05,
        },
        {
            "desc": "Aggressive",
            "max_depth": 8,
            "eta": 0.18,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
            "reg_lambda": 0.5,
            "gamma": 0.0,
        },
    ]

    # EQUITY: shallow trees, stronger regularization (noisy, mean-reverting)
    if mt == "EQUITY":
        return [
            {
                "desc": "Equity Conservative",
                "max_depth": 3,
                "eta": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 5.0,
                "reg_lambda": 3.0,
                "gamma": 0.15,
            },
            {
                "desc": "Equity Balanced",
                "max_depth": 5,
                "eta": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 3.0,
                "reg_lambda": 1.5,
                "gamma": 0.1,
            },
            {
                "desc": "Equity Aggressive",
                "max_depth": 6,
                "eta": 0.12,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 2.0,
                "reg_lambda": 1.0,
                "gamma": 0.05,
            },
        ]

    # FUTURES: trend-driven, allow deeper trees & faster learning
    if mt == "FUTURES":
        return [
            {
                "desc": "Futures Conservative",
                "max_depth": 5,
                "eta": 0.06,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 3.0,
                "reg_lambda": 1.5,
                "gamma": 0.05,
            },
            {
                "desc": "Futures Balanced",
                "max_depth": 7,
                "eta": 0.10,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": 2.0,
                "reg_lambda": 1.0,
                "gamma": 0.03,
            },
            {
                "desc": "Futures Aggressive",
                "max_depth": 8,
                "eta": 0.16,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 0.8,
                "gamma": 0.0,
            },
        ]

    # OPTIONS: more non-linear; keep eta smaller, allow depth
    if mt == "OPTIONS":
        return [
            {
                "desc": "Options Conservative",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 4.0,
                "reg_lambda": 2.5,
                "gamma": 0.1,
            },
            {
                "desc": "Options Balanced",
                "max_depth": 6,
                "eta": 0.08,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": 2.0,
                "reg_lambda": 1.5,
                "gamma": 0.08,
            },
            {
                "desc": "Options Aggressive",
                "max_depth": 7,
                "eta": 0.12,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "gamma": 0.03,
            },
        ]

    # CRYPTO: volatile; deeper trees but regularization
    if mt == "CRYPTO":
        return [
            {
                "desc": "Crypto Conservative",
                "max_depth": 4,
                "eta": 0.06,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 4.0,
                "reg_lambda": 2.0,
                "gamma": 0.1,
            },
            {
                "desc": "Crypto Balanced",
                "max_depth": 6,
                "eta": 0.10,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": 2.0,
                "reg_lambda": 1.2,
                "gamma": 0.05,
            },
            {
                "desc": "Crypto Aggressive",
                "max_depth": 7,
                "eta": 0.14,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "gamma": 0.02,
            },
        ]

    return generic


# ---------------------------------------------------------------------
# Core Trainer
# ---------------------------------------------------------------------


class SmartTrainer:
    """
    Institutional-grade trainer.

    Upgrades:
    - Basket training (multi-symbol) for EQUITY
    - Walk-forward TimeSeriesSplit cross-validation
    - Market-specific hyperparameters
    - Threshold optimisation (best prob cutoff)
    - Final retrain on full dataset with tuned num_boost_round
    - Metadata export alongside XGBoost model
    """

    @staticmethod
    def _fetch_and_build_for_symbol(
        symbol: str,
        market_type: str,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """
        Fetches raw data for a single symbol and applies feature engineering.
        Returns dataframe with 'target' + features, or None if insufficient.
        """
        print(f"[ML] Fetching last {lookback_days} days of data for {symbol}...")
        df = DataFetcher.fetch(symbol, market=market_type, days=lookback_days)

        if df is None or len(df) < 200:
            print(f"[ML WARNING] Insufficient data for {symbol}. Skipping this symbol.")
            return None

        print(f"[ML] Generating 22-factor feature set for {symbol}...")
        df_feat = FeatureEngineering.build(df)
        df_feat = df_feat.dropna()

        if len(df_feat) < 100:
            print(f"[ML WARNING] Not enough post-feature rows for {symbol}. Skipping this symbol.")
            return None

        # Tag with symbol so we can optionally use it as a feature
        df_feat["__symbol__"] = symbol
        return df_feat

    # --------------------------------------------------------------
    # Main training function
    # --------------------------------------------------------------

    @staticmethod
    def train_and_save(
        market_type: str,
        symbols,
        lookback_days: int,
        target_col: str = "target",
    ):
        """
        Train and save an edge model.

        `symbols` can be:
            - "RELIANCE"
            - ["RELIANCE", "HDFCBANK", "ICICIBANK", ...]
        """

        # Normalise symbols to list form
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = list(symbols)

        pretty_symbol = ", ".join(symbol_list)
        print(f"\n--- SMART TRAINING: {market_type} ({pretty_symbol}) ---")

        # 1. Fetch & build all symbols
        all_frames: List[pd.DataFrame] = []
        for sym in symbol_list:
            try:
                df_feat = SmartTrainer._fetch_and_build_for_symbol(sym, market_type, lookback_days)
                if df_feat is not None:
                    all_frames.append(df_feat)
            except Exception:
                print(f"[ML ERROR] Exception while processing {sym}:")
                traceback.print_exc()

        if not all_frames:
            print(f"[ML ERROR] No usable data for any symbol in {pretty_symbol}. Skipping.")
            return

        df = pd.concat(all_frames, axis=0)
        df = df.sort_index()

        # Encode symbol as numeric feature if multiple symbols
        if "__symbol__" in df.columns:
            df["symbol_code"] = df["__symbol__"].astype("category").cat.codes
            df = df.drop(columns=["__symbol__"])

        if df is None or len(df) < 200:
            print(f"[ML ERROR] Insufficient combined data for {pretty_symbol}. Skipping.")
            return

        # Separate features / target
        if target_col not in df.columns:
            print(f"[ML ERROR] Target column '{target_col}' missing. Skipping.")
            return

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        # ------------------------------------------------------------------
        # 2. Decide whether to use Walk-Forward CV
        # ------------------------------------------------------------------
        n_samples = len(df)
        use_cv = n_samples >= 240  # below this, 80/20 split is safer

        configs = _get_market_configs(market_type)

        best_mean_acc = 0.0
        best_std_acc = 0.0
        best_cfg: Dict[str, Any] | None = None
        best_num_boost_round = 0
        best_threshold = 0.5

        print("[ML] Auto-Tuning Hyperparameters...")

        for cfg in configs:
            # Base XGBoost params
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": int(cfg["max_depth"]),
                "eta": float(cfg["eta"]),
                "subsample": float(cfg["subsample"]),
                "colsample_bytree": float(cfg["colsample_bytree"]),
                "min_child_weight": float(cfg.get("min_child_weight", 1.0)),
                "lambda": float(cfg.get("reg_lambda", 1.0)),
                "gamma": float(cfg.get("gamma", 0.0)),
                "seed": 42,
                "tree_method": "hist",
            }

            desc = cfg.get("desc", "Config")

            fold_accs: List[float] = []
            fold_thresholds: List[float] = []
            fold_best_iters: List[int] = []

            # ------------------------------
            # 2A. Walk-Forward CV
            # ------------------------------
            if use_cv:
                # dynamic splits: at least 2, at most 5, ~60 bars per fold minimum
                max_splits = min(5, max(2, n_samples // 60))
                tscv = TimeSeriesSplit(n_splits=max_splits)

                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dval = xgb.DMatrix(X_val, label=y_val)

                    booster = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=800,
                        evals=[(dtrain, "train"), (dval, "eval")],
                        early_stopping_rounds=50,
                        verbose_eval=False,
                    )

                    preds = booster.predict(dval)

                    # Threshold optimisation between 0.45–0.55
                    thresholds = np.linspace(0.45, 0.55, 11)
                    best_acc_fold = 0.0
                    best_thr_fold = 0.5

                    for thr in thresholds:
                        pred_labels = (preds > thr).astype(int)
                        acc = float(np.mean(pred_labels == y_val))
                        if acc > best_acc_fold:
                            best_acc_fold = acc
                            best_thr_fold = float(thr)

                    fold_accs.append(best_acc_fold * 100.0)
                    fold_thresholds.append(best_thr_fold)
                    fold_best_iters.append(booster.best_iteration or 0)

            # ------------------------------
            # 2B. Simple 80/20 split
            # ------------------------------
            else:
                split_idx = int(n_samples * 0.80)
                train_df = df.iloc[:split_idx]
                test_df = df.iloc[split_idx:]

                X_train = train_df.drop(columns=[target_col])
                y_train = train_df[target_col].astype(int)

                X_val = test_df.drop(columns=[target_col])
                y_val = test_df[target_col].astype(int)

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=800,
                    evals=[(dtrain, "train"), (dval, "eval")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )

                preds = booster.predict(dval)

                thresholds = np.linspace(0.45, 0.55, 11)
                best_acc_fold = 0.0
                best_thr_fold = 0.5

                for thr in thresholds:
                    pred_labels = (preds > thr).astype(int)
                    acc = float(np.mean(pred_labels == y_val))
                    if acc > best_acc_fold:
                        best_acc_fold = acc
                        best_thr_fold = float(thr)

                fold_accs.append(best_acc_fold * 100.0)
                fold_thresholds.append(best_thr_fold)
                fold_best_iters.append(booster.best_iteration or 0)

            # Aggregate metrics for this config
            mean_acc = float(np.mean(fold_accs))
            std_acc = float(np.std(fold_accs))
            mean_thr = float(np.mean(fold_thresholds))
            mean_iter = int(np.mean([i for i in fold_best_iters if i is not None] or [100]))

            print(
                f"   > {desc} CV Accuracy: {mean_acc:.2f}% "
                f"(std: {std_acc:.2f}, avg_iter: {mean_iter}, best_thr: {mean_thr:.3f})"
            )

            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_std_acc = std_acc
                best_cfg = cfg
                best_num_boost_round = max(50, mean_iter)
                best_threshold = mean_thr

        # ------------------------------------------------------------------
        # 3. Final Verdict + Retrain on Full Data
        # ------------------------------------------------------------------

        if not best_cfg:
            print("[ML ERROR] No valid configuration found. Skipping save.")
            return

        desc = best_cfg.get("desc", "Unknown")
        print(f"[ML] Winner: {desc} Model with {best_mean_acc:.2f}% mean CV accuracy.")

        if best_mean_acc < 52.0:
            print("[ML WARNING] Model is weak (near random). Features might need improvement.")
        elif best_mean_acc > 80.0:
            print("[ML WARNING] Accuracy suspiciously high. Check for look-ahead bias.")
        else:
            print("[ML SUCCESS] Model ready for deployment.")

        # Retrain final model on the full dataset using tuned params
        final_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": int(best_cfg["max_depth"]),
            "eta": float(best_cfg["eta"]),
            "subsample": float(best_cfg["subsample"]),
            "colsample_bytree": float(best_cfg["colsample_bytree"]),
            "min_child_weight": float(best_cfg.get("min_child_weight", 1.0)),
            "lambda": float(best_cfg.get("reg_lambda", 1.0)),
            "gamma": float(best_cfg.get("gamma", 0.0)),
            "seed": 42,
            "tree_method": "hist",
        }

        dtrain_full = xgb.DMatrix(X, label=y)
        final_booster = xgb.train(
            final_params,
            dtrain_full,
            num_boost_round=int(best_num_boost_round),
            evals=[(dtrain_full, "train")],
            verbose_eval=False,
        )

        # ------------------------------------------------------------------
        # 4. Save Model + Metadata
        # ------------------------------------------------------------------
        model_filename = f"{market_type.lower()}_edge.json"
        model_path = os.path.join(MODEL_DIR, model_filename)

        final_booster.save_model(model_path)
        print(f"[ML] Saved model to: {model_path}")

        meta = {
            "market_type": market_type.upper(),
            "symbols": symbol_list,
            "lookback_days": lookback_days,
            "n_samples": int(n_samples),
            "feature_names": final_booster.feature_names,
            "best_config": {
                "desc": desc,
                "max_depth": int(best_cfg["max_depth"]),
                "eta": float(best_cfg["eta"]),
                "subsample": float(best_cfg["subsample"]),
                "colsample_bytree": float(best_cfg["colsample_bytree"]),
                "min_child_weight": float(best_cfg.get("min_child_weight", 1.0)),
                "reg_lambda": float(best_cfg.get("reg_lambda", 1.0)),
                "gamma": float(best_cfg.get("gamma", 0.0)),
            },
            "cv_mean_accuracy": round(best_mean_acc, 3),
            "cv_std_accuracy": round(best_std_acc, 3),
            "best_threshold": round(best_threshold, 4),
            "best_num_boost_round": int(best_num_boost_round),
        }

        meta_filename = f"{market_type.lower()}_edge_meta.json"
        meta_path = os.path.join(MODEL_DIR, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[ML] Saved metadata to: {meta_path}")


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def auto_train_all():
    print("====== REELIOO SMART AUTO-TRAIN (v7.0) ======")

    # ========= EQUITY =========
    equity_basket = [
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "INFY",
        "TCS",
        "LT",
        "SBIN",
    ]
    try:
        SmartTrainer.train_and_save("EQUITY", equity_basket, lookback_days=730)
    except Exception:
        traceback.print_exc()

    # ========= FUTURES =========
    try:
        SmartTrainer.train_and_save("FUTURES", "NIFTY", lookback_days=1000)
    except Exception:
        traceback.print_exc()

    # ========= OPTIONS =========
    try:
        SmartTrainer.train_and_save("OPTIONS", "NIFTY", lookback_days=365)
    except Exception:
        traceback.print_exc()

    # ========= CRYPTO =========
    try:
        SmartTrainer.train_and_save("CRYPTO", "BTC-INR", lookback_days=365)
    except Exception:
        traceback.print_exc()

    print("\n====== TRAINING COMPLETE ======")


if __name__ == "__main__":
    auto_train_all()
