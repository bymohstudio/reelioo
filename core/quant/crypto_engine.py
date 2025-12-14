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

    # MUST MATCH TRAINING
    ML_FEATURES = [
        'rsi', 'squeeze', 'vol_z',
        'trend_strength', 'atr_ratio',
        'bb_width', 'body_size'
    ]

    # ---------------------------------------------------------
    # 1. LOAD MODEL + META
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
        data['trend_strength'] = (ema9 - ema21) / close * 100

        # ATR Ratio
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - close.shift()).abs(),
            (data['low'] - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        data['atr_ratio'] = atr / close

        # Candle Body
        data['body_size'] = (close - data['open']).abs() / close

        return data.fillna(0)

    # ---------------------------------------------------------
    # 3. MAIN ANALYSIS
    # ---------------------------------------------------------
    def analyze(self, df: pd.DataFrame, trade_style: str = "SWING"):
        if df is None or df.empty:
            raise ValueError("Empty dataframe")

        # A. Indicators
        features = self._calculate_indicators(df)
        last = features.iloc[-1]

        # B. Load model + meta
        model, meta = self._load_artifacts()

        # -----------------------------------------------------
        # C. ML PROBABILITY
        # -----------------------------------------------------
        ml_prob = 0.5
        if model:
            try:
                row = pd.DataFrame([last])[self.ML_FEATURES].astype(float)
                dmat = xgb.DMatrix(row)
                ml_prob = float(model.predict(dmat)[0])
            except Exception as e:
                log.error(f"Prediction failed: {e}")

        # -----------------------------------------------------
        # 🔑 THRESHOLD ENFORCEMENT (THIS IS WHAT YOU ASKED)
        # -----------------------------------------------------
        best_threshold = 0.65  # fallback
        if meta and isinstance(meta, dict):
            best_threshold = meta.get("best_threshold", best_threshold)

        if ml_prob < best_threshold:
            return SimpleNamespace(
                score=50,
                bias="NEUTRAL",
                entry=None,
                target1=None,
                target2=None,
                target3=None,
                stop=None,
                rr_ratio=0.0,
                expected_duration="Low Confidence",
                regime="Low Confidence",
                regime_color="gray",
                whale_zscore=round(float(last.get("vol_z", 0)), 2),
                whale_label="Normal",
                sentiment_headline="Low confidence ML signal",
                sentiment_score=0.0,
                top_features=[],
                model_metrics=meta.get("metrics", {}) if meta else {}
            )

        # -----------------------------------------------------
        # D. SCORING
        # -----------------------------------------------------
        score = ml_prob * 100
        trend_val = float(last.get('trend_strength', 0))
        score += trend_val * 10
        score = max(5, min(99, score))

        if score >= 60:
            bias = "LONG"
        elif score <= 40:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        # -----------------------------------------------------
        # E. LEVELS
        # -----------------------------------------------------
        price = float(df['close'].iloc[-1])
        atr = price * 0.02 if last.get('atr_ratio', 0) == 0 else price * last['atr_ratio']

        if bias == "LONG":
            stop = price - atr * 1.5
            t1 = price + atr * 1.5
            t2 = price + atr * 2.5
            t3 = price + atr * 5
        elif bias == "SHORT":
            stop = price + atr * 1.5
            t1 = price - atr * 1.5
            t2 = price - atr * 2.5
            t3 = price - atr * 5
        else:
            stop = price
            t1 = t2 = t3 = price

        rr = abs(t1 - price) / abs(price - stop) if price != stop else 0

        metrics = meta.get("metrics", {}) if meta else {}

        return SimpleNamespace(
            score=round(score, 0),
            bias=bias,
            entry=round(price, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            stop=round(stop, 4),
            rr_ratio=round(rr, 2),
            expected_duration="6–72 HOURS",
            regime="Trending" if abs(trend_val) > 0.2 else "Ranging",
            regime_color="green" if trend_val > 0 else "red",
            whale_zscore=round(float(last.get("vol_z", 0)), 2),
            whale_label="High Volume" if abs(last.get("vol_z", 0)) > 1.5 else "Normal",
            sentiment_headline="ML + Quant Confirmed",
            sentiment_score=round(ml_prob * 100, 1),
            top_features=[],
            model_metrics={
                "auc": metrics.get("auc"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "best_threshold": best_threshold,
                "trained_on": meta.get("trained_date") if meta else None
            }
        )
