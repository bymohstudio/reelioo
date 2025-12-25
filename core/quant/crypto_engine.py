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

        # 1. Load XGBoost (Independent Try/Catch)
        try:
            if os.path.exists(self.PATHS['xgb_long']):
                self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
                self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
        except Exception as e:
            log.error(f"XGB Load Error: {e}")

        # 2. Load LightGBM (Independent Try/Catch)
        try:
            if os.path.exists(self.PATHS['lgb_long']):
                self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
                self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])
        except Exception as e:
            log.error(f"LGB Load Error: {e}")

        # 3. Load CatBoost (Independent Try/Catch)
        try:
            if os.path.exists(self.PATHS['cat_long']):
                self.models['cat_long'] = CatBoostClassifier()
                self.models['cat_long'].load_model(self.PATHS['cat_long'])
                self.models['cat_short'] = CatBoostClassifier()
                self.models['cat_short'].load_model(self.PATHS['cat_short'])
        except Exception as e:
            log.error(f"CatBoost Load Error: {e}")

        return self.models

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY"):
        df = generate_features(df)
        last = df.iloc[-1]

        # Prepare row for ML
        row_df = pd.DataFrame([last])[FEATURES].astype(float)

        models = self._load_models()

        # If absolutely no models loaded, return neutral
        if not models:
            return self._neutral_result(last['close'], "System Booting")

        # 1. GET RAW PROBABILITIES (Safe Prediction Logic)
        pL_vals = []
        pS_vals = []

        try:
            # XGBoost Prediction
            if 'xgb_long' in models and 'xgb_short' in models:
                dmat = xgb.DMatrix(row_df)
                pL_vals.append(float(models['xgb_long'].predict(dmat)[0]))
                pS_vals.append(float(models['xgb_short'].predict(dmat)[0]))

            # LightGBM Prediction
            if 'lgb_long' in models and 'lgb_short' in models:
                pL_vals.append(float(models['lgb_long'].predict(row_df)[0]))
                pS_vals.append(float(models['lgb_short'].predict(row_df)[0]))

            # CatBoost Prediction
            if 'cat_long' in models and 'cat_short' in models:
                pL_vals.append(float(models['cat_long'].predict_proba(row_df)[0][1]))
                pS_vals.append(float(models['cat_short'].predict_proba(row_df)[0][1]))

            # Calculate Averages (Ensemble)
            if not pL_vals:
                return self._neutral_result(last['close'], "Model Error")

            pL = (sum(pL_vals) / len(pL_vals)) * 100
            pS = (sum(pS_vals) / len(pS_vals)) * 100

        except Exception as e:
            log.error(f"Prediction Calculation Error: {e}")
            return self._neutral_result(last['close'], "Calc Error")

        # 2. THE CASINO "HOUSE RULES" (Tier 4 Execution Logic)

        bias = "HOLD"
        # Default score is the max probability so user sees "45%" instead of "0%"
        score = max(pL, pS)

        # Context Variables
        rsi = last['rsi_14']
        vwap_dist = last['vwap_dist']  # +ve means Price > VWAP (Premium), -ve means Discount
        cvd_div = last.get('cvd_divergence', 0)
        liq_sweep = last.get('liq_sweep', 0)

        # Trend Physics
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']
        price = last['close']

        is_uptrend = (price > ema_20) and (ema_20 > ema_50)
        is_downtrend = (price < ema_20) and (ema_20 < ema_50)

        # --- DYNAMIC THRESHOLD (UX FIX) ---
        # If SCALP (High Risk), we lower the bar to 60%.
        # If DAY/SWING (Safe), we keep it at 65% (Sniper).
        if trade_style == "SCALP":
            CONF_THRESH = 60.0
        else:
            CONF_THRESH = 65.0

        # --- LONG LOGIC ---
        if pL > CONF_THRESH:
            # Rule 1: Don't Buy the Top (RSI check)
            if rsi < 70:
                # Rule 2: Value Check (Prefer buying below or near VWAP, or if momentum is huge)
                if vwap_dist < 0.02:  # Don't buy if > 2% above VWAP
                    # Bonus Confidence if we swept liquidity or have CVD Div
                    bonus = 0
                    if liq_sweep == 1: bonus += 5
                    if cvd_div == 1: bonus += 5

                    if (pL + bonus) >= CONF_THRESH:
                        bias = "LONG"
                        score = pL + bonus

        # --- SHORT LOGIC ---
        elif pS > CONF_THRESH:
            # Rule 1: Don't Short the Bottom
            if rsi > 30:
                # Rule 2: Value Check (Prefer shorting above or near VWAP)
                if vwap_dist > -0.02:  # Don't short if > 2% below VWAP
                    # Bonus Confidence
                    bonus = 0
                    if liq_sweep == -1: bonus += 5
                    if cvd_div == -1: bonus += 5

                    if (pS + bonus) >= CONF_THRESH:
                        bias = "SHORT"
                        score = pS + bonus

        # --- SAFETY VALVE (Conflict Resolution) ---
        # If we triggered logic but trend opposes, or signals conflict, revert to HOLD
        if bias == "LONG" and not is_uptrend and trade_style != "SCALP":
            bias = "HOLD"
        if bias == "SHORT" and not is_downtrend and trade_style != "SCALP":
            bias = "HOLD"

        # 3. TRADE MANAGEMENT (Risk Engine)
        atr = float(last.get('atr_14', price * 0.01))

        # Dynamic Multipliers
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

        # 4. EXPLAINABILITY (Why did we take this?)
        drivers = []
        if abs(vwap_dist) > 0.01:
            drivers.append({"feature": "Value", "desc": "VWAP DEVIATION", "importance": 90})
        if liq_sweep != 0:
            drivers.append({"feature": "Trap", "desc": "LIQUIDITY SWEEP", "importance": 95})
        if cvd_div != 0:
            drivers.append({"feature": "Whale", "desc": "CVD DIVERGENCE", "importance": 85})

        # Fill remaining with generic features if needed
        if not drivers:
            drivers.append({"feature": "Trend", "desc": "MOMENTUM ALIGNMENT", "importance": 80})

        dist = abs(t1 - price)

        return SimpleNamespace(
            bias=bias,
            score=int(min(99, round(score))),
            entry=price,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t1 + (dist * 0.5), 4),
            target3=round(t1 + dist, 4),
            rr_ratio=round(tgt_mult / stop_mult, 2),
            expected_duration=duration,
            regime="LIQUIDITY RUN" if liq_sweep != 0 else "TREND",
            regime_color="green" if bias == "LONG" else "red" if bias == "SHORT" else "gray",
            whale_zscore=round(float(last.get('whale_z', 0)), 2),
            whale_label="Whale Active" if abs(last.get('whale_z', 0)) > 2.0 else "Normal",
            top_features=drivers
        )

    def _neutral_result(self, price, reason="Neutral"):
        return SimpleNamespace(
            bias="HOLD", score=50, entry=price, stop=price, target1=price, target2=price,
            target3=price, rr_ratio=0, expected_duration="--", regime=reason,
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[]
        )