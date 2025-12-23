# core/quant/crypto_engine.py

from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
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
            self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
            self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
            self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
            self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])
            self.models['cat_long'] = CatBoostClassifier()
            self.models['cat_long'].load_model(self.PATHS['cat_long'])
            self.models['cat_short'] = CatBoostClassifier()
            self.models['cat_short'].load_model(self.PATHS['cat_short'])
        except Exception as e:
            log.warning(f"Model Load Error: {e}")
        return self.models

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY"):

        df = generate_features(df)
        last = df.iloc[-1]
        row_df = pd.DataFrame([last])[FEATURES].astype(float)

        models = self._load_models()

        # ML PREDICTION: PROBABILITY OF VOLATILITY EXPLOSION (REGIME)
        # pL = Prob of Upside Explosion
        # pS = Prob of Downside Explosion
        pL = (float(models['xgb_long'].predict(xgb.DMatrix(row_df))[0]) +
              float(models['lgb_long'].predict(row_df)[0]) +
              float(models['cat_long'].predict_proba(row_df)[0][1])) / 3 * 100

        pS = (float(models['xgb_short'].predict(xgb.DMatrix(row_df))[0]) +
              float(models['lgb_short'].predict(row_df)[0]) +
              float(models['cat_short'].predict_proba(row_df)[0][1])) / 3 * 100

        # === HYBRID EXECUTION LOGIC (TIER 4) ===
        # 1. ML says "Storm Coming" (High Prob)
        # 2. Physics says "Wind Blowing East" (EMA Trend)

        bias = "HOLD"
        score = 50

        # Trend Physics
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']
        price = last['close']

        is_uptrend = (price > ema_20) and (ema_20 > ema_50)
        is_downtrend = (price < ema_20) and (ema_20 < ema_50)

        # Thresholds
        CONF_THRESH = 65.0

        if pL > CONF_THRESH and is_uptrend:
            bias = "LONG"
            score = pL
        elif pS > CONF_THRESH and is_downtrend:
            bias = "SHORT"
            score = pS
        else:
            # If signals conflict (e.g., ML says Up but Trend is Down), we HOLD.
            # This is the "Safety Valve" preventing fakeouts.
            bias = "HOLD"
            score = max(pL, pS) if max(pL, pS) < 60 else 55  # Show mild interest but no execute

        # Stop/Target Logic (Volatility Based)
        atr = float(last.get('atr_14', price * 0.01))

        if trade_style == "SCALP":
            stop_mult, tgt_mult = 1.0, 1.5
            duration = "15m - 2h"
        elif trade_style == "SWING":
            stop_mult, tgt_mult = 2.5, 4.0
            duration = "1 - 3 Days"
        else:  # DAY
            stop_mult, tgt_mult = 2.0, 3.0
            duration = "4h - 24h"

        if bias == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * tgt_mult)
        elif bias == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * tgt_mult)
        else:
            stop, t1 = price, price

        dist = abs(t1 - price)
        regime = "MOMENTUM" if bias != "HOLD" else "CHOP/RANGE"
        regime_color = "green" if bias == "LONG" else "red" if bias == "SHORT" else "gray"

        whale_z = float(last.get('whale_z', 0))
        whale_label = "High Flow" if abs(whale_z) > 1.5 else "Normal"

        # Explainability
        drivers = []
        if float(last.get('cvd_slope', 0)) > 0:
            drivers.append({"feature": "Order Flow", "desc": "BUY PRESSURE (CVD)", "importance": 95})
        if float(last.get('ttm_squeeze', 0)) > 0:
            drivers.append({"feature": "Volatility", "desc": "TTM SQUEEZE", "importance": 90})
        if is_uptrend:
            drivers.append({"feature": "Trend", "desc": "EMA ALIGNMENT", "importance": 85})

        drivers.sort(key=lambda x: x['importance'], reverse=True)

        return SimpleNamespace(
            bias=bias,
            score=int(round(score)),
            entry=price,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t1 + (dist * 0.5), 4),
            target3=round(t1 + dist, 4),
            rr_ratio=round(tgt_mult / stop_mult, 2),
            expected_duration=duration,
            regime=regime,
            regime_color=regime_color,
            whale_zscore=round(whale_z, 2),
            whale_label=whale_label,
            top_features=drivers[:3]
        )