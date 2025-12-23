import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import logging
from django.conf import settings
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.models = {}

        # --- PATH FIX: USE DJANGO SETTINGS ---
        # This ensures we look in "ROOT/core/quant/ml_models" no matter what.
        self.MODEL_DIR = os.path.join(settings.BASE_DIR, "core", "quant", "ml_models")

        # Debug Log: Tells you exactly where it is looking
        print(f"🔍 BACKTEST ENGINE LOOKING IN: {self.MODEL_DIR}")

        self.PATHS = {
            "xgb_long": os.path.join(self.MODEL_DIR, "xgb_long.json"),
            "xgb_short": os.path.join(self.MODEL_DIR, "xgb_short.json"),
            "lgb_long": os.path.join(self.MODEL_DIR, "lgb_long.txt"),
            "lgb_short": os.path.join(self.MODEL_DIR, "lgb_short.txt"),
            "cat_long": os.path.join(self.MODEL_DIR, "cat_long.cbm"),
            "cat_short": os.path.join(self.MODEL_DIR, "cat_short.cbm"),
        }
        self._load_models()

    def _load_models(self):
        try:
            # Check if directory exists first
            if not os.path.exists(self.MODEL_DIR):
                log.error(f"❌ DIRECTORY MISSING: {self.MODEL_DIR}")
                print(f"❌ DIRECTORY MISSING: {self.MODEL_DIR}")
                return

            if os.path.exists(self.PATHS['xgb_long']):
                self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
                self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
            else:
                print(f"❌ MODEL MISSING: {self.PATHS['xgb_long']}")

            if os.path.exists(self.PATHS['lgb_long']):
                self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
                self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])

            if os.path.exists(self.PATHS['cat_long']):
                self.models['cat_long'] = CatBoostClassifier()
                self.models['cat_long'].load_model(self.PATHS['cat_long'])
                self.models['cat_short'] = CatBoostClassifier()
                self.models['cat_short'].load_model(self.PATHS['cat_short'])

        except Exception as e:
            log.error(f"Backtest Model Load Error: {e}")
            print(f"❌ EXCEPTION LOADING MODELS: {e}")

    def run(self, trade_style="INTRADAY"):
        if self.df is None or self.df.empty:
            return self._empty_result()

        df = generate_features(self.df.copy())

        # Check if models loaded successfully
        if not self.models or 'xgb_long' not in self.models:
            log.error("Backtest Error: Models not loaded. Check file paths.")
            return self._empty_result()

        # --- RESILIENT HUNTER SETTINGS (Matching Live Engine) ---
        if trade_style == "SCALP":
            min_conf = 55.0
            min_eff = 0.05
            stop_mult, tgt_mult = 1.0, 1.5
        elif trade_style == "SWING":
            min_conf = 65.0
            min_eff = 0.08
            stop_mult, tgt_mult = 2.5, 4.0
        else:  # DAY / INTRADAY
            min_conf = 60.0
            min_eff = 0.09
            stop_mult = 2.0
            tgt_mult = 3.0

            # Predict Batch
        try:
            X = df[FEATURES].astype(float)
            dmat = xgb.DMatrix(X)
            xl = self.models['xgb_long'].predict(dmat)
            xs = self.models['xgb_short'].predict(dmat)
            ll = self.models['lgb_long'].predict(X)
            ls = self.models['lgb_short'].predict(X)
            cl = self.models['cat_long'].predict_proba(X)[:, 1]
            cs = self.models['cat_short'].predict_proba(X)[:, 1]

            df['ens_long'] = ((xl + ll + cl) / 3) * 100
            df['ens_short'] = ((xs + ls + cs) / 3) * 100
        except Exception as e:
            log.error(f"Prediction Error: {e}")
            return self._empty_result()

        # Simulate
        position = None
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        start_idx = 50
        for i in range(start_idx, len(df)):
            curr = df.iloc[i]
            price = curr['close']
            atr = curr.get('atr_14', price * 0.01)

            # Exit
            if position:
                res = None
                if position == 'LONG':
                    if curr['low'] <= stop_loss:
                        res, pnl = "LOSS", -1.0
                    elif curr['high'] >= take_profit:
                        res, pnl = "WIN", (take_profit - entry_price) / entry_price * 100
                elif position == 'SHORT':
                    if curr['high'] >= stop_loss:
                        res, pnl = "LOSS", -1.0
                    elif curr['low'] <= take_profit:
                        res, pnl = "WIN", (entry_price - take_profit) / entry_price * 100

                if res:
                    self.trades.append({"result": res, "pnl": round(pnl, 2), "entry": entry_price})
                    position = None
                    continue

            # Entry
            if not position:
                eff = curr.get('efficiency_ratio', 0)
                vol = curr.get('volatility_slope', 0)

                if (eff > min_eff) or (vol > 0.2):
                    p_l = curr['ens_long']
                    p_s = curr['ens_short']

                    if p_l > min_conf and p_l > p_s + 5:
                        position = 'LONG'
                        entry_price = price
                        stop_loss = price - (atr * stop_mult)
                        take_profit = price + (atr * tgt_mult)
                    elif p_s > min_conf and p_s > p_l + 5:
                        position = 'SHORT'
                        entry_price = price
                        stop_loss = price + (atr * stop_mult)
                        take_profit = price - (atr * tgt_mult)

        return self._generate_stats()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0:
            return self._empty_result()

        wins = [t for t in self.trades if t['result'] == 'WIN']
        wr = (len(wins) / total * 100) if total > 0 else 0

        gross_profit = sum(t['pnl'] for t in wins)
        losses = [t for t in self.trades if t['pnl'] < 0]
        gross_loss = abs(sum(t['pnl'] for t in losses))

        pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

        return {
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "total_trades": total,
            "trades_log": self.trades[-20:]
        }

    def _empty_result(self):
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "trades_log": []
        }