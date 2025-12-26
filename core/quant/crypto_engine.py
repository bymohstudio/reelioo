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

    # Paths to your trained models
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
            # --- 1. XGBOOST (Force CPU Mode) ---
            self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
            self.models['xgb_long'].set_param({"predictor": "cpu_predictor", "nthread": 1})

            self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
            self.models['xgb_short'].set_param({"predictor": "cpu_predictor", "nthread": 1})

            # --- 2. LIGHTGBM (Naturally CPU) ---
            self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
            self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])

            # --- 3. CATBOOST (Force CPU Mode) ---
            self.models['cat_long'] = CatBoostClassifier()
            self.models['cat_long'].load_model(self.PATHS['cat_long'])

            self.models['cat_short'] = CatBoostClassifier()
            self.models['cat_short'].load_model(self.PATHS['cat_short'])

        except Exception as e:
            pass
        return self.models

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY"):
        df = generate_features(df)
        last = df.iloc[-1]

        # Prepare row for ML
        row_df = pd.DataFrame([last])[FEATURES].astype(float)

        models = self._load_models()
        if not models:
            return self._neutral_result(last['close'], "System Booting")

        # 1. GET RAW PROBABILITIES (The "Math")
        pL, pS = 0.0, 0.0
        try:
            # XGBoost DMatrix (Force CPU usage)
            dmat = xgb.DMatrix(row_df)

            pL = (float(models['xgb_long'].predict(dmat)[0]) +
                  float(models['lgb_long'].predict(row_df)[0]) +
                  float(models['cat_long'].predict_proba(row_df)[0][1])) / 3 * 100

            pS = (float(models['xgb_short'].predict(dmat)[0]) +
                  float(models['lgb_short'].predict(row_df)[0]) +
                  float(models['cat_short'].predict_proba(row_df)[0][1])) / 3 * 100
        except Exception as e:
            log.error(f"Prediction Error: {e}")
            return self._neutral_result(last['close'], "Model Error")

        # 2. THE CASINO "HOUSE RULES"
        bias = "HOLD"
        score = max(pL, pS)

        # Context Variables
        rsi = last['rsi_14']
        vwap_dist = last['vwap_dist']
        liq_sweep = last.get('liq_sweep', 0)
        vol_slope = last.get('volatility_slope', 0)

        # Trend Physics
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']
        price = last['close']

        is_uptrend = (price > ema_20) and (ema_20 > ema_50)
        is_downtrend = (price < ema_20) and (ema_20 < ema_50)

        # --- DATA-DRIVEN SNIPER THRESHOLDS ---
        LONG_THRESH = 70.0
        SHORT_THRESH = 99.0

        if trade_style == "SCALP":
            LONG_THRESH -= 5.0

        # --- LONG LOGIC ---
        if pL > LONG_THRESH:
            if rsi < 75:
                if vwap_dist < 0.04:
                    if is_uptrend or vol_slope > 0.1 or liq_sweep == 1:
                        bias = "LONG"
                        score = pL
                        if liq_sweep == 1: score += 5

        # --- SHORT LOGIC ---
        elif pS > SHORT_THRESH:
            if rsi > 25:
                if vwap_dist > -0.04:
                    if is_downtrend:
                        bias = "SHORT"
                        score = pS
                        if liq_sweep == -1: score += 5

        # --- SAFETY VALVE ---
        if bias == "LONG" and not is_uptrend and trade_style != "SCALP":
            if liq_sweep != 1: bias = "HOLD"

        if bias == "SHORT" and not is_downtrend:
            bias = "HOLD"

        # 3. TRADE MANAGEMENT (Risk Engine)
        atr = float(last.get('atr_14', price * 0.01))

        if trade_style == "SCALP":
            stop_mult, tgt_mult = 1.0, 1.5
            duration = "15m - 2h"
        elif trade_style == "SWING":
            stop_mult, tgt_mult = 2.5, 4.0
            duration = "1 - 3 Days"
        else:  # DAY
            stop_mult, tgt_mult = 1.5, 1.5
            duration = "4h - 24h"

        calc_direction = bias
        if bias == "HOLD":
            if pL >= pS:
                calc_direction = "LONG"
            else:
                calc_direction = "SHORT"

        if calc_direction == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * tgt_mult)
        elif calc_direction == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * tgt_mult)
        else:
            stop, t1 = price, price

        # 4. EXPLAINABILITY & NARRATIVE
        drivers = []
        if vol_slope > 0.1:
            drivers.append({"feature": "Energy", "desc": "VOLATILITY SPIKE", "importance": 95})
        if abs(vwap_dist) < 0.01:
            drivers.append({"feature": "Value", "desc": "FAIR VALUE ENTRY", "importance": 85})
        if liq_sweep != 0:
            drivers.append({"feature": "Trap", "desc": "STOP HUNT", "importance": 90})

        if not drivers:
            drivers.append({"feature": "Trend", "desc": "AI MOMENTUM", "importance": 80})

        dist = abs(t1 - price)

        # HUMAN NARRATIVE GENERATOR
        narrative = self._generate_narrative(bias, drivers, rsi, is_uptrend, liq_sweep)

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
            top_features=drivers,
            narrative=narrative  # NEW FIELD
        )

    def _generate_narrative(self, bias, drivers, rsi, is_uptrend, liq_sweep):
        if bias == "HOLD":
            if not is_uptrend: return "Market structure is broken. Protecting capital."
            if rsi > 70: return "Asset is overextended. Waiting for pullback."
            return "Volume is too low for a safe entry. Staying passive."

        main_driver = drivers[0]['desc'] if drivers else "MOMENTUM"

        if liq_sweep != 0:
            return "Whales swept liquidity stops. Reversal imminent."
        if "VOLATILITY" in main_driver:
            return "Volatility expansion detected. Breakout likely."
        if "FAIR VALUE" in main_driver:
            return "Price reclaimed VWAP baseline. Institutional entry zone."

        return "Trend alignment confirmed with strong volume."

    def _neutral_result(self, price, reason="Neutral"):
        return SimpleNamespace(
            bias="HOLD", score=50, entry=price, stop=price, target1=price, target2=price,
            target3=price, rr_ratio=0, expected_duration="--", regime=reason,
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[],
            narrative="System initializing data streams..."
        )