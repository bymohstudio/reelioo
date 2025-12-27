# core/quant/crypto_engine.py

from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import logging
from core.quant.ml_training.feature_engineering import generate_features, FEATURES

# --- Flow Bridge Import ---
from core.quant.flow_bridge import get_btc_flow_snapshot

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
        "flow_model": os.path.join(MODEL_DIR, "flow_model.json"),
    }

    def __init__(self):
        self.models = {}

    def _load_models(self):
        if self.models: return self.models
        try:
            # --- 1. XGBOOST ---
            self.models['xgb_long'] = xgb.Booster(model_file=self.PATHS['xgb_long'])
            self.models['xgb_long'].set_param({"predictor": "cpu_predictor", "nthread": 1})

            self.models['xgb_short'] = xgb.Booster(model_file=self.PATHS['xgb_short'])
            self.models['xgb_short'].set_param({"predictor": "cpu_predictor", "nthread": 1})

            # Load Flow Model (Silent fail allowed)
            if os.path.exists(self.PATHS['flow_model']):
                self.models['flow_model'] = xgb.Booster(model_file=self.PATHS['flow_model'])
                self.models['flow_model'].set_param({"predictor": "cpu_predictor", "nthread": 1})

            # --- 2. LIGHTGBM ---
            self.models['lgb_long'] = lgb.Booster(model_file=self.PATHS['lgb_long'])
            self.models['lgb_short'] = lgb.Booster(model_file=self.PATHS['lgb_short'])

            # --- 3. CATBOOST ---
            self.models['cat_long'] = CatBoostClassifier()
            self.models['cat_long'].load_model(self.PATHS['cat_long'])

            self.models['cat_short'] = CatBoostClassifier()
            self.models['cat_short'].load_model(self.PATHS['cat_short'])

        except Exception as e:
            log.error(f"Model Loading Error: {e}")
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

        # 1. GET RAW PRICE PROBABILITIES
        pL, pS = 0.0, 0.0
        try:
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

        # ---------------------------------------------------------
        # 🟢 FLOW INTELLIGENCE LAYER
        # ---------------------------------------------------------
        flow_score = 0.5  # Default Neutral
        flow_veto = False
        flow_narrative = ""

        if trade_style != "SCALP" and 'flow_model' in models:
            try:
                flow_data = get_btc_flow_snapshot()
                if flow_data is not None:
                    dmat_flow = xgb.DMatrix(flow_data, feature_names=["liq_pressure", "funding_z"])
                    flow_score = float(models['flow_model'].predict(dmat_flow)[0])
            except Exception as ex:
                log.warning(f"Flow Inference Failed: {ex}")

        # ---------------------------------------------------------
        # 🔴 LOGIC: CALIBRATION & VETO
        # ---------------------------------------------------------

        # A) CALIBRATION: Boost confidence if flow aligns
        if pL > pS:
            if flow_score > 0.70: pL += 5.0
            if flow_score < 0.30: pL -= 10.0
        else:
            if flow_score < 0.30: pS += 5.0
            if flow_score > 0.70: pS -= 10.0

        # B) SELECT BIAS
        bias = "HOLD"
        score = max(pL, pS)

        # Context Variables
        rsi = last['rsi_14']
        vwap_dist = last['vwap_dist']
        liq_sweep = last.get('liq_sweep', 0)
        vol_slope = last.get('volatility_slope', 0)
        ema_20 = last['ema_20']
        ema_50 = last['ema_50']
        price = last['close']

        # TREND DEFINITIONS
        is_uptrend = (price > ema_20) and (ema_20 > ema_50)
        # We relax Short requirements: If Price < EMA20, it's weak enough to dump (even if EMA20 > EMA50)
        is_weakness = (price < ema_20)

        # --- THRESHOLDS (UPDATED) ---
        LONG_THRESH = 70.0
        SHORT_THRESH = 70.0  # UNLOCKED (Was 99.0)

        if trade_style == "SCALP":
            LONG_THRESH -= 5.0
            SHORT_THRESH -= 5.0

        if pL > LONG_THRESH:
            if rsi < 75 and vwap_dist < 0.04:
                if is_uptrend or vol_slope > 0.1 or liq_sweep == 1:
                    bias = "LONG"
                    score = pL
                    if liq_sweep == 1: score += 5

        elif pS > SHORT_THRESH:
            if rsi > 25 and vwap_dist > -0.04:
                # REPLACED: Strict `is_downtrend` with relaxed `is_weakness`
                if is_weakness or vol_slope > 0.2:
                    bias = "SHORT"
                    score = pS
                    if liq_sweep == -1: score += 5

        # Safety Valves
        if bias == "LONG" and not is_uptrend and trade_style != "SCALP":
            if liq_sweep != 1: bias = "HOLD"

        if bias == "SHORT" and not is_weakness:
            # If we are trying to short but price is ABOVE EMA20, it's too risky (Bear Trap)
            bias = "HOLD"

        # C) FLOW VETO
        if trade_style != "SCALP" and bias != "HOLD":
            if bias == "LONG" and flow_score < 0.25:
                bias = "HOLD"
                flow_veto = True
                flow_narrative = "Global BTC liquidations contradict Long setup."
            elif bias == "SHORT" and flow_score > 0.75:
                bias = "HOLD"
                flow_veto = True
                flow_narrative = "Global BTC liquidations contradict Short setup."

        # 3. TRADE MANAGEMENT
        atr = float(last.get('atr_14', price * 0.01))

        if trade_style == "SCALP":
            stop_mult, tgt_mult, duration = 1.0, 1.5, "15m - 2h"
        elif trade_style == "SWING":
            stop_mult, tgt_mult, duration = 2.5, 4.0, "1 - 3 Days"
        else:  # DAY
            stop_mult, tgt_mult, duration = 1.5, 1.5, "4h - 24h"

        calc_dir = "LONG" if (bias == "LONG" or (bias == "HOLD" and pL >= pS)) else "SHORT"

        if calc_dir == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * tgt_mult)
        else:
            stop = price + (atr * stop_mult)
            t1 = price - (atr * tgt_mult)

        # 4. EXPLAINABILITY
        drivers = []
        if vol_slope > 0.1: drivers.append({"feature": "Energy", "desc": "VOLATILITY SPIKE", "importance": 95})
        if abs(vwap_dist) < 0.01: drivers.append({"feature": "Value", "desc": "FAIR VALUE ENTRY", "importance": 85})
        if liq_sweep != 0: drivers.append({"feature": "Trap", "desc": "STOP HUNT", "importance": 90})

        if abs(flow_score - 0.5) > 0.3:
            lbl = "BULLISH FLOW" if flow_score > 0.5 else "BEARISH FLOW"
            drivers.append({"feature": "On-Chain", "desc": lbl, "importance": 88})

        if not drivers: drivers.append({"feature": "Trend", "desc": "AI MOMENTUM", "importance": 80})

        dist = abs(t1 - price)
        narrative = self._generate_narrative(bias, drivers, rsi, is_uptrend, liq_sweep, flow_veto, flow_narrative)

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
            narrative=narrative,
            flow_score=round(flow_score, 2)
        )

    def _generate_narrative(self, bias, drivers, rsi, is_uptrend, liq_sweep, flow_veto, flow_narrative):
        if flow_veto:
            return f"Trade Vetoed: {flow_narrative}"

        if bias == "HOLD":
            return "Market noise. Waiting for clear structure."

        main_driver = drivers[0]['desc'] if drivers else "MOMENTUM"

        if liq_sweep != 0: return "Whales swept liquidity stops. Reversal imminent."
        if "VOLATILITY" in main_driver: return "Volatility expansion detected. Fast move expected."
        if "FAIR VALUE" in main_driver: return "Price reclaimed VWAP baseline. Institutional entry zone."
        if "FLOW" in main_driver: return "Smart money flows are aggressively supporting this direction."

        return "Trend alignment confirmed with strong volume."

    def _neutral_result(self, price, reason="Neutral"):
        return SimpleNamespace(
            bias="HOLD", score=50, entry=price, stop=price, target1=price, target2=price,
            target3=price, rr_ratio=0, expected_duration="--", regime=reason,
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[],
            narrative="System initializing data streams...", flow_score=0.5
        )