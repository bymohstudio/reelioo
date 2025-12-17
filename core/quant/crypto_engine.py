from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import xgboost as xgb
import os
import json
import logging
from .feature_engineering import generate_features, FEATURES

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LONG_PATH = os.path.join(BASE_DIR, "ml_models", "long_model.json")
    SHORT_PATH = os.path.join(BASE_DIR, "ml_models", "short_model.json")

    def _load_models(self):
        m_long, m_short = None, None
        try:
            if os.path.exists(self.LONG_PATH):
                m_long = xgb.Booster()
                m_long.load_model(self.LONG_PATH)
            if os.path.exists(self.SHORT_PATH):
                m_short = xgb.Booster()
                m_short.load_model(self.SHORT_PATH)
        except:
            pass
        return m_long, m_short

    def analyze(self, df: pd.DataFrame, trade_style: str = "SWING"):
        if df.empty: raise ValueError("Empty Data")

        # 1. Features
        df = generate_features(df)
        last = df.iloc[-1]

        # 2. Get Predictions
        m_long, m_short = self._load_models()
        prob_long, prob_short = 0.0, 0.0

        if m_long and m_short:
            try:
                row = pd.DataFrame([last])[FEATURES].astype(float)
                dmat = xgb.DMatrix(row)
                prob_long = float(m_long.predict(dmat)[0]) * 100
                prob_short = float(m_short.predict(dmat)[0]) * 100
            except:
                pass

        # 3. Decision Logic (Real-World Calibrated)
        # 65% is strong for this model type. 70% is Sniper.
        min_conf = 70.0 if trade_style == "SCALP" else 65.0

        final_bias = "NEUTRAL"
        score = 50.0
        regime = "WAIT"
        color = "gray"

        # 4. Filters (Efficiency Check - THE FIX IS HERE)
        # Must be efficient OR breaking out (Vol expansion)
        is_clean = last['efficiency_ratio'] > 0.15 or last['volatility_slope'] > 0.1

        if is_clean:
            if prob_long > min_conf and prob_long > (prob_short + 10):
                final_bias = "LONG"
                score = prob_long
                regime = "BULLISH SNIPER"
                color = "green"
            elif prob_short > min_conf and prob_short > (prob_long + 10):
                final_bias = "SHORT"
                score = prob_short
                regime = "BEARISH SNIPER"
                color = "red"


        # 5. Levels (ATR Based)
        price = float(last['close'])
        atr = float(last.get('atr_14', price * 0.01))

        # MATCHING TRAINING LOGIC (1.5 R:R)
        stop_mult = 1.5
        target_mult = 2.25  # 1.5 * 1.5 = 2.25

        if final_bias == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * target_mult)
        elif final_bias == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * target_mult)
        else:
            stop = t1 = price

        dist = abs(t1 - price)
        t2 = t1 + (dist * 0.5)
        t3 = t1 + (dist * 1.0)

        return SimpleNamespace(
            score=int(score),
            bias=final_bias,
            entry=price,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=1.5,  # Updated display
            expected_duration="1-6 Hours" if trade_style == "SCALP" else "1-3 Days",
            regime=regime,
            regime_color=color,
            whale_zscore=round(float(last.get('vol_z', 0)), 2),
            whale_label="High Vol" if abs(last.get('vol_z', 0)) > 2 else "Normal",
            top_features=[]
        )