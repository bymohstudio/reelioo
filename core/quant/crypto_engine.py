from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import json
import logging
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

    PATHS = {
        "xgb_long": os.path.join(MODEL_DIR, "xgb_long.json"),
        "xgb_short": os.path.join(MODEL_DIR, "xgb_short.json"),
        "lgb_long": os.path.join(MODEL_DIR, "lgb_long.txt"),
        "lgb_short": os.path.join(MODEL_DIR, "lgb_short.txt"),
        "cat_long": os.path.join(MODEL_DIR, "cat_long.cbm"),
        "cat_short": os.path.join(MODEL_DIR, "cat_short.cbm"),
    }

    def __init__(self):
        self.models = {}

    def _load_models(self):
        if self.models: return self.models
        try:
            # XGBoost
            self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
            self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
            # LightGBM
            self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
            self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])
            # CatBoost
            self.models['cat_long'] = CatBoostClassifier()
            self.models['cat_long'].load_model(self.PATHS['cat_long'])
            self.models['cat_short'] = CatBoostClassifier()
            self.models['cat_short'].load_model(self.PATHS['cat_short'])
        except Exception as e:
            log.warning(f"Model Load Error (Partial?): {e}")
        return self.models

    def analyze(self, df: pd.DataFrame, trade_style: str = "SWING"):
        if df.empty: raise ValueError("Empty Data")

        # 1. Feature Engineering
        df = generate_features(df)
        last = df.iloc[-1]
        row_df = pd.DataFrame([last])[FEATURES].astype(float)

        # 2. Ensemble Prediction
        models = self._load_models()
        prob_long, prob_short = 0.0, 0.0

        # --- VOTING LOGIC ---
        if 'xgb_long' in models and 'lgb_long' in models:
            # Longs
            xl = float(models['xgb_long'].predict(xgb.DMatrix(row_df))[0])
            ll = float(models['lgb_long'].predict(row_df)[0])
            cl = float(models['cat_long'].predict_proba(row_df)[0][1])
            prob_long = ((xl + ll + cl) / 3.0) * 100

            # Shorts
            xs = float(models['xgb_short'].predict(xgb.DMatrix(row_df))[0])
            ls = float(models['lgb_short'].predict(row_df)[0])
            cs = float(models['cat_short'].predict_proba(row_df)[0][1])
            prob_short = ((xs + ls + cs) / 3.0) * 100

        # 3. DYNAMIC TRADING LOGIC (The "Retail Friendly" Part)
        # Adjust thresholds based on User Preference

        if trade_style == "SCALP":
            # SCALP: Wants frequent trades, smaller moves.
            min_conf = 60.0  # Lower threshold (Retail wants action)
            min_efficiency = 0.05  # Accepts messier markets
            duration = "15m - 2h"
            regime_prefix = "SCALP"

        elif trade_style == "SWING":
            # SWING: Wants precision, bigger moves.
            min_conf = 70.0  # High threshold
            min_efficiency = 0.15  # Needs cleaner trends
            duration = "1 - 3 Days"
            regime_prefix = "SWING"

        else:  # INTRADAY
            min_conf = 65.0
            min_efficiency = 0.10
            duration = "4h - 24h"
            regime_prefix = "DAY"

        # 4. Final Decision
        final_bias = "NEUTRAL"
        score = 50.0
        regime = "WAIT"
        color = "gray"

        # Check Market Quality (Efficiency Filter)
        # We use the dynamic 'min_efficiency' from above
        is_clean = last['efficiency_ratio'] > min_efficiency or last['volatility_slope'] > 0.1

        if is_clean:
            if prob_long > min_conf and prob_long > (prob_short + 5):
                final_bias = "LONG"
                score = prob_long
                regime = f"{regime_prefix} BUY"
                color = "green"
            elif prob_short > min_conf and prob_short > (prob_long + 5):
                final_bias = "SHORT"
                score = prob_short
                regime = f"{regime_prefix} SELL"
                color = "red"

        # 5. Risk Levels (ATR Based)
        price = float(last['close'])
        atr = float(last.get('atr_14', price * 0.01))

        # Adjust Risk/Reward for Scalping
        if trade_style == "SCALP":
            stop_mult, target_mult = 1.0, 1.5  # Tighter stops for scalping
        else:
            stop_mult, target_mult = 1.5, 2.5  # Wider room for swings

        if final_bias == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * target_mult)
        elif final_bias == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * target_mult)
        else:
            stop = t1 = price

        dist = abs(t1 - price)

        return SimpleNamespace(
            score=int(score),
            bias=final_bias,
            entry=price,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t1 + (dist * 0.5), 4),
            target3=round(t1 + dist, 4),
            rr_ratio=round(target_mult / stop_mult, 1),
            expected_duration=duration,
            regime=regime,
            regime_color=color,
            whale_zscore=round(float(last.get('vol_z', 0)), 2),
            whale_label="High Vol" if abs(last.get('vol_z', 0)) > 2 else "Normal",
            top_features=[]
        )