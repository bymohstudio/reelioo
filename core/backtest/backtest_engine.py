import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import logging
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    Ensemble Backtester (XGB + LGB + CAT).
    Validates the exact 3-model voting logic used in the live environment.
    """
    # Adjust path to find the ml_models folder relative to this file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

    PATHS = {
        "xgb_long": os.path.join(MODEL_DIR, "xgb_long.json"),
        "xgb_short": os.path.join(MODEL_DIR, "xgb_short.json"),
        "lgb_long": os.path.join(MODEL_DIR, "lgb_long.txt"),
        "lgb_short": os.path.join(MODEL_DIR, "lgb_short.txt"),
        "cat_long": os.path.join(MODEL_DIR, "cat_long.cbm"),
        "cat_short": os.path.join(MODEL_DIR, "cat_short.cbm"),
    }

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.models = {}

        # Load models immediately
        self._load_models()

    def _load_models(self):
        """Loads all available models for the ensemble."""
        try:
            # XGBoost
            if os.path.exists(self.PATHS['xgb_long']):
                self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
                self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])

            # LightGBM
            if os.path.exists(self.PATHS['lgb_long']):
                self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
                self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])

            # CatBoost
            if os.path.exists(self.PATHS['cat_long']):
                self.models['cat_long'] = CatBoostClassifier()
                self.models['cat_long'].load_model(self.PATHS['cat_long'])
                self.models['cat_short'] = CatBoostClassifier()
                self.models['cat_short'].load_model(self.PATHS['cat_short'])

        except Exception as e:
            log.error(f"Backtest Model Load Error: {e}")

    def _generate_predictions(self):
        """
        Vectorized prediction: Runs models on the entire dataframe at once.
        This is much faster than running prediction inside the loop.
        """
        if self.df.empty: return self.df

        # 1. Generate Features (Same as Live)
        df_feat = generate_features(self.df)
        X = df_feat[FEATURES]

        # Initialize probabilities
        self.df['ens_long'] = 0.0
        self.df['ens_short'] = 0.0

        # XGBoost Prediction
        if 'xgb_long' in self.models:
            dmat = xgb.DMatrix(X)
            self.df['xgb_l'] = self.models['xgb_long'].predict(dmat)
            self.df['xgb_s'] = self.models['xgb_short'].predict(dmat)
        else:
            self.df['xgb_l'] = self.df['xgb_s'] = 0.0

        # LightGBM Prediction
        if 'lgb_long' in self.models:
            self.df['lgb_l'] = self.models['lgb_long'].predict(X)
            self.df['lgb_s'] = self.models['lgb_short'].predict(X)
        else:
            self.df['lgb_l'] = self.df['lgb_s'] = 0.0

        # CatBoost Prediction (Slower, but accurate)
        if 'cat_long' in self.models:
            self.df['cat_l'] = self.models['cat_long'].predict_proba(X)[:, 1]
            self.df['cat_s'] = self.models['cat_short'].predict_proba(X)[:, 1]
        else:
            self.df['cat_l'] = self.df['cat_s'] = 0.0

        # ENSEMBLE AVERAGE (Voting Logic)
        self.df['ens_long'] = (self.df['xgb_l'] + self.df['lgb_l'] + self.df['cat_l']) / 3.0 * 100
        self.df['ens_short'] = (self.df['xgb_s'] + self.df['lgb_s'] + self.df['cat_s']) / 3.0 * 100

        # Add Filter Columns needed for logic
        self.df['efficiency_ratio'] = df_feat['efficiency_ratio']
        self.df['volatility_slope'] = df_feat['volatility_slope']

        return self.df

    def run(self, trade_style="INTRADAY"):
        """
        Simulates the strategy with Dynamic Logic based on trade_style.
        """
        # 1. Pre-calculate all AI scores
        self._generate_predictions()

        # 2. Set Dynamic Thresholds (Must match crypto_engine.py)
        if trade_style == "SCALP":
            min_conf = 60.0
            min_efficiency = 0.05
            stop_mult, target_mult = 1.0, 1.5
        elif trade_style == "SWING":
            min_conf = 70.0
            min_efficiency = 0.15
            stop_mult, target_mult = 1.5, 2.5
        else:  # INTRADAY
            min_conf = 65.0
            min_efficiency = 0.10
            stop_mult, target_mult = 1.5, 2.25

        position = None
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        start_idx = 50  # Skip warm-up

        for i in range(start_idx, len(self.df)):
            curr = self.df.iloc[i]
            price = float(curr['close'])

            # Simple ATR calc if missing
            tr = max(curr['high'] - curr['low'], abs(curr['high'] - curr['close']))
            atr = tr if tr > 0 else price * 0.01

            # --- 1. EXIT LOGIC ---
            if position:
                exit_price = 0.0
                result = ""

                if position == 'LONG':
                    if curr['low'] <= stop_loss:
                        exit_price = stop_loss
                        result = "LOSS"
                    elif curr['high'] >= take_profit:
                        exit_price = take_profit
                        result = "WIN"

                elif position == 'SHORT':
                    if curr['high'] >= stop_loss:
                        exit_price = stop_loss
                        result = "LOSS"
                    elif curr['low'] <= take_profit:
                        exit_price = take_profit
                        result = "WIN"

                if exit_price != 0.0:
                    pnl = (exit_price - entry_price) / entry_price if position == 'LONG' else (
                                                                                                          entry_price - exit_price) / entry_price

                    self.trades.append({
                        "type": position,
                        "style": trade_style,
                        "entry": round(entry_price, 4),
                        "exit": round(exit_price, 4),
                        "pnl": round(pnl * 100, 2),
                        "result": result,
                        "date": str(curr.name)
                    })
                    position = None
                    continue

            # --- 2. ENTRY LOGIC (AI ENSEMBLE) ---
            if position is None:
                # A. Filter Checks
                is_clean = curr['efficiency_ratio'] > min_efficiency or curr['volatility_slope'] > 0.1
                if not is_clean: continue

                # B. Vote Checks
                p_long = curr['ens_long']
                p_short = curr['ens_short']

                if p_long > min_conf and p_long > (p_short + 5):
                    position = 'LONG'
                    stop_loss = price - (atr * stop_mult)
                    take_profit = price + (atr * target_mult)
                    entry_price = price

                elif p_short > min_conf and p_short > (p_long + 5):
                    position = 'SHORT'
                    stop_loss = price + (atr * stop_mult)
                    take_profit = price - (atr * target_mult)
                    entry_price = price

        return self._generate_stats()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []}

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = round((len(wins) / total) * 100, 1)
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9

        return {
            "win_rate": win_rate,
            "profit_factor": pf,
            "total_trades": total,
            "trades_log": self.trades[::-1][:50]
        }