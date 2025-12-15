from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    MARKET_TYPE = "CRYPTO"

    # Path setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "crypto_edge.json")
    META_PATH = os.path.join(BASE_DIR, "ml_models", "edge_meta.json")

    _MODEL = None
    _META = None

    ML_FEATURES = [
        'rsi', 'squeeze', 'vol_z',
        'trend_strength', 'atr_ratio',
        'bb_width', 'body_size'
    ]

    # ---------------------------------------------------------
    # 1. LOAD MODEL
    # ---------------------------------------------------------
    @classmethod
    def _load_artifacts(cls):
        if cls._META is None and os.path.exists(cls.META_PATH):
            try:
                with open(cls.META_PATH, 'r') as f:
                    cls._META = json.load(f)
            except Exception as e:
                log.error(f"Meta load failed: {e}")

        if cls._MODEL is None and os.path.exists(cls.MODEL_PATH):
            try:
                booster = xgb.Booster()
                booster.load_model(cls.MODEL_PATH)
                cls._MODEL = booster
            except Exception as e:
                log.error(f"Model load failed: {e}")

        return cls._MODEL, cls._META

    # ---------------------------------------------------------
    # 2. FEATURE ENGINE
    # ---------------------------------------------------------
    def _calculate_indicators(self, df: pd.DataFrame):
        data = df.copy()
        close = data['close']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        data['squeeze'] = (std20 * 4) / sma20
        data['bb_width'] = data['squeeze']

        # Volume Z
        vol_mean = data['volume'].rolling(20).mean()
        vol_std = data['volume'].rolling(20).std()
        data['vol_z'] = (data['volume'] - vol_mean) / vol_std

        # Trend Strength
        ema9 = close.ewm(span=9).mean()
        ema21 = close.ewm(span=21).mean()
        ema50 = close.ewm(span=50).mean()

        data['trend_strength'] = (ema9 - ema21) / close * 100
        data['ema_9'] = ema9
        data['ema_21'] = ema21
        data['ema_50'] = ema50

        # ATR Ratio
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - close.shift()).abs(),
            (data['low'] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        data['atr_ratio'] = atr / close
        data['atr'] = atr

        # Candle Body
        data['body_size'] = (close - data['open']).abs() / close

        return data.fillna(0)

    # ---------------------------------------------------------
    # 3. MICRO-STRUCTURE LAYER
    # ---------------------------------------------------------
    def _analyze_micro_structure(self, last_row):
        close = float(last_row['close'])
        ema21 = float(last_row['ema_21'])
        ema50 = float(last_row['ema_50'])

        if close > ema21 and ema21 > ema50:
            bias = "LONG"
        elif close < ema21 and ema21 < ema50:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        return bias

    # ---------------------------------------------------------
    # 4. MAIN ANALYSIS
    # ---------------------------------------------------------
    def analyze(self, df: pd.DataFrame, trade_style: str = "SWING"):
        if df is None or df.empty:
            raise ValueError("Empty dataframe")

        features = self._calculate_indicators(df)
        last = features.iloc[-1]
        model, meta = self._load_artifacts()

        # --- LAYER 1: PRIMARY INTELLIGENCE ---
        ml_prob = 0.5

        if model:
            try:
                row = pd.DataFrame([last])[self.ML_FEATURES].astype(float)
                dmat = xgb.DMatrix(row)
                ml_prob = float(model.predict(dmat)[0])
            except Exception as e:
                log.error(f"Prediction failed: {e}")

        raw_score = ml_prob * 100

        # Determine Bias
        if raw_score >= 60:
            primary_bias = "LONG"
            engine_mode = "PRIMARY_ML"
        elif raw_score <= 40:
            primary_bias = "SHORT"
            engine_mode = "PRIMARY_ML"
        else:
            primary_bias = "NEUTRAL"
            engine_mode = "NEUTRAL"

        # --- LAYER 2: LOGIC ROUTING ---
        final_bias = primary_bias

        # 🚀 NORMALIZATION FIX:
        # If it's SHORT (e.g. 20%), the confidence is actually 80%.
        # If it's LONG (e.g. 80%), the confidence is 80%.
        if raw_score < 50:
            conviction_score = 100 - raw_score
        else:
            conviction_score = raw_score

        regime_label = "AI_SWING"
        regime_color = "gray"
        duration = "Wait"
        signal_strength = "NONE"

        if engine_mode == "PRIMARY_ML":
            final_bias = primary_bias
            regime_label = "STRONG_TREND"
            regime_color = "green"
            duration = "Swing (1-3 Days)"
            signal_strength = "STRONG"

        else:
            # Primary is Neutral -> Check Micro-Structure
            micro_bias = self._analyze_micro_structure(last)

            if micro_bias != "NEUTRAL":
                final_bias = micro_bias
                # Speculative fixed score
                conviction_score = 60.0
                regime_label = "MICRO_SCALP"
                regime_color = "yellow"
                duration = "Scalp (15m - 4h)"
                signal_strength = "SPECULATIVE"
            else:
                final_bias = "NEUTRAL"
                conviction_score = 50.0
                regime_label = "CONSOLIDATION"
                duration = "Stand Aside"
                signal_strength = "NONE"

        # --- LAYER 3: LEVEL GENERATION ---
        price = float(df['close'].iloc[-1])
        atr = float(last['atr'])
        if atr == 0: atr = price * 0.01

        if signal_strength == "STRONG":
            stop_mult, t1_mult = 2.0, 2.0
        elif signal_strength == "SPECULATIVE":
            stop_mult, t1_mult = 1.0, 1.5
        else:
            stop_mult, t1_mult = 1.0, 1.0

        if final_bias == "LONG":
            stop = price - (atr * stop_mult)
            t1 = price + (atr * t1_mult)
            t2 = price + (atr * t1_mult * 2)
            t3 = price + (atr * t1_mult * 3)
        elif final_bias == "SHORT":
            stop = price + (atr * stop_mult)
            t1 = price - (atr * t1_mult)
            t2 = price - (atr * t1_mult * 2)
            t3 = price - (atr * t1_mult * 3)
        else:
            stop = price
            t1 = t2 = t3 = price

        rr = abs(t1 - price) / abs(price - stop) if price != stop else 0.0

        return SimpleNamespace(
            # 🚀 RETURN NORMALIZED CONVICTION SCORE (e.g. 78% instead of 22%)
            score=round(conviction_score, 0),

            bias=final_bias,
            entry=round(price, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            stop=round(stop, 4),
            rr_ratio=round(rr, 2),
            expected_duration=duration,
            regime=regime_label.replace("_", " "),
            regime_color=regime_color,
            whale_zscore=round(float(last.get("vol_z", 0)), 2),
            whale_label="High Vol" if abs(last.get("vol_z", 0)) > 2 else "Normal",
            sentiment_headline=f"{regime_label.replace('_', ' ')} Detected",
            sentiment_score=round(conviction_score, 1),
            top_features=[],
            model_metrics=meta.get("metrics", {}) if meta else {}
        )