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
        if self.models:
            return self.models
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

    def analyze(self, df: pd.DataFrame, trade_style: str = "SWING"):

        df = generate_features(df)
        last = df.iloc[-1]
        row_df = pd.DataFrame([last])[FEATURES].astype(float)

        models = self._load_models()

        # Prediction Logic
        if 'xgb_long' in models:
            pL = (float(models['xgb_long'].predict(xgb.DMatrix(row_df))[0]) +
                  float(models['lgb_long'].predict(row_df)[0]) +
                  float(models['cat_long'].predict_proba(row_df)[0][1])) / 3 * 100

            pS = (float(models['xgb_short'].predict(xgb.DMatrix(row_df))[0]) +
                  float(models['lgb_short'].predict(row_df)[0]) +
                  float(models['cat_short'].predict_proba(row_df)[0][1])) / 3 * 100
        else:
            pL, pS = 50.0, 50.0

        # ---- Thresholds ----
        if trade_style == "SCALP":
            min_conf = 58.0
            min_eff = 0.08
            duration = "15m - 2h"
        elif trade_style == "SWING":
            min_conf = 60.0
            min_eff = 0.08
            duration = "1 - 3 Days"
        else:
            min_conf = 60.0
            min_eff = 0.08
            duration = "4h - 24h"

        bias = "NEUTRAL"
        score = max(pL, pS)

        eff = float(last['efficiency_ratio'])
        vol_slope = float(last['volatility_slope'])

        market_ok = (eff > min_eff) or (vol_slope > 0.1)

        if market_ok:
            if pL > min_conf and pL > (pS + 5):
                bias = "LONG"
                score = pL
            elif pS > min_conf and pS > (pL + 5):
                bias = "SHORT"
                score = pS

        price = float(last['close'])
        atr = float(last.get('atr_14', price * 0.01))

        stop_mult = 1.0 if trade_style == "SCALP" else 1.5
        tgt_mult = 1.5 if trade_style == "SCALP" else 2.5

        if bias == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * tgt_mult)
        elif bias == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * tgt_mult)
        else:
            stop = price
            t1 = price

        dist = abs(t1 - price)

        # ---- Regime & Whale context ----
        if score >= 60:
            regime = "ACTIVE"
            regime_color = "green" if bias == "LONG" else "red"
        else:
            regime = "WAIT"
            regime_color = "gray"

        whale_z = float(last.get('vol_z', 0))
        whale_label = "High Vol" if abs(whale_z) > 2 else "Normal"

        # ---- 6. CALCULATE LOGIC VECTORS (Updated Names) ----
        drivers = []

        # Driver A: Efficiency (Trend Quality)
        eff = float(last.get('efficiency_ratio', 0))
        if eff > 0.1:
            drivers.append({
                "feature": "Trend Efficiency",
                "desc": "CLEAN PRICE PATH",  # Matches Screenshot
                "importance": min(eff * 200, 95)
            })

        # Driver B: Whale Volume (Institutional Action)
        whale_z = float(last.get('vol_z', 0))
        if abs(whale_z) > 1.2:
            drivers.append({
                "feature": "Whale Volume",
                "desc": "WHALE ACCUMULATION",  # Matches Screenshot
                "importance": min(abs(whale_z) * 20, 92)
            })

        # Driver C: Momentum (Trend Strength)
        trend = float(last.get('trend_strength', 0))
        if abs(trend) > 0.5:
            drivers.append({
                "feature": "Momentum Trend",
                "desc": "TREND MOMENTUM",  # Matches Screenshot
                "importance": 85.0
            })

        # Driver D: Squeeze (Volatility Compression)
        if float(last.get('ttm_squeeze', 0)) > 0:
            drivers.append({
                "feature": "TTM Squeeze",
                "desc": "VOLATILITY SQUEEZE",
                "importance": 88.0
            })

        # Driver E: Velocity (High Return)
        if float(last.get('ret_3', 0)) > 0.05:  # > 5% move
            drivers.append({
                "feature": "Velocity",
                "desc": "VELOCITY HIGH",  # Matches Screenshot
                "importance": 90.0
            })

        # Sort by importance and take Top 3
        drivers.sort(key=lambda x: x['importance'], reverse=True)
        top_vectors = drivers[:3]

        return SimpleNamespace(
            # ... (Existing fields like bias, score, entry, etc.) ...
            score=int(round(score)),
            bias=bias,
            entry=price,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t1 + (dist * 0.5), 4),
            target3=round(t1 + dist, 4),
            rr_ratio=round(tgt_mult / stop_mult, 1),
            expected_duration=duration,
            regime=regime,
            regime_color=regime_color,
            whale_zscore=round(whale_z, 2),
            whale_label=whale_label,
            top_features=top_vectors  # This is critical
        )