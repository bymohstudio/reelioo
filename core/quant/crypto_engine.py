# core/quant/crypto_engine.py

from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


def cap(x, limit=3.0):
    """Bounds a value to prevent outliers from hijacking the score."""
    return max(-limit, min(limit, x))


class CryptoQuantEngine:
    """
    REELIOO QUANT PHYSICS ENGINE (v5.3 – Price Fix)

    - Bounded Factors & Regime Awareness (Production Safety)
    - WATCH State: Yellow Color, Bias = "WATCH"
    - UI Fix: 'entry' always returns CURRENT PRICE (so header isn't $--)
    - Targets (SL/TP) remain 0.0 unless Confirmed.
    """

    def __init__(self):
        self.SIGMOID_K = 0.45
        self.CONFIRMATION_THRESH = 70
        log.info("🚀 QuantPhysicsEngine v5.3 (Price Fix) Initialized")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        try:
            df = generate_features(df)
            last = df.iloc[-1]
        except Exception as e:
            return self._neutral_result(0, f"Data Error: {e}")

        price = float(last["close"])

        # ------------------------------------------------------------------
        # 1. REGIME DETECTION
        # ------------------------------------------------------------------
        er = float(last.get("efficiency_ratio", 0.5))
        regime_label = "TRENDING" if er > 0.4 else "CHOPPY"

        # ------------------------------------------------------------------
        # 2. ALPHA FACTORS (PHYSICS)
        # ------------------------------------------------------------------
        # Trend
        ema_diff = cap(last.get("ema_diff", 0) * 100)
        rsi_z = cap((last.get("rsi_14", 50) - 50) / 15)
        trend_alpha = (ema_diff * 1.5 + rsi_z) * (1.0 if regime_label == "TRENDING" else 0.5)
        trend_alpha = cap(trend_alpha)

        # Whale
        whale_z = cap(float(last.get("whale_z", 0)))

        # Reversion
        vwap_z = cap(last.get("vwap_dist", 0) * 100)
        reversion_alpha = -vwap_z * (2.0 if regime_label == "CHOPPY" else 1.2)
        reversion_alpha = cap(reversion_alpha)

        # Events
        event_alpha = 0.0
        if int(last.get("liq_sweep", 0)) == 1:
            event_alpha += 2.0
        elif int(last.get("liq_sweep", 0)) == -1:
            event_alpha -= 2.0

        if int(last.get("cvd_divergence", 0)) == 1:
            event_alpha += 1.5
        elif int(last.get("cvd_divergence", 0)) == -1:
            event_alpha -= 1.5
        event_alpha = cap(event_alpha)

        # ------------------------------------------------------------------
        # 3. PROBABILITY
        # ------------------------------------------------------------------
        raw_alpha = trend_alpha + whale_z + reversion_alpha + event_alpha
        final_probability = self._sigmoid(raw_alpha)

        # Stability Dampener (Fakeout Filter)
        vol_slope = float(last.get("volatility_slope", 0))
        if vol_slope > 0.25 and abs(whale_z) < 1.0:
            final_probability = 50 + (final_probability - 50) * 0.75

        # ------------------------------------------------------------------
        # 4. INITIAL BIAS & SCORE
        # ------------------------------------------------------------------
        if final_probability > 55:
            bias = "LONG"
            score = final_probability
        elif final_probability < 45:
            bias = "SHORT"
            score = 100 - final_probability
        else:
            bias = "HOLD"
            score = 50

        # Score Compression
        score = 50 + (score - 50) * 0.85

        # ------------------------------------------------------------------
        # 5. VISUAL STATE & LOGIC
        # ------------------------------------------------------------------
        thresh = self.CONFIRMATION_THRESH - (5 if trade_style == "SCALP" else 0)

        regime_color = "gray"
        display_bias = bias

        # FIX: Entry is ALWAYS price (so header shows price)
        entry = price

        # Trade Levels default to 0.0 (Hidden)
        stop, t1, t2, t3 = 0.0, 0.0, 0.0, 0.0
        expected_duration = "--"

        # A. CONFIRMED TRADE (GREEN/RED)
        if score >= thresh:
            regime_color = "green" if bias == "LONG" else "red"
            display_bias = bias  # Keep LONG/SHORT

            # Calculate Targets ONLY if Confirmed
            atr = float(last.get("atr_14", price * 0.01))
            stop_mult, tgt_mult = (1.0, 1.5) if trade_style == "SCALP" else (1.5, 3.0)
            direction = 1 if bias == "LONG" else -1

            stop = price - direction * atr * stop_mult
            t1 = price + direction * atr * tgt_mult
            t2 = t1 + (abs(t1 - price) * 0.5) * direction
            t3 = t1 + abs(t1 - price) * direction

            expected_duration = "4h - 24h"

        # B. WATCHING STATE (YELLOW)
        elif score >= 60:
            regime_color = "yellow"
            display_bias = "WATCH"
            # Targets remain 0.0, but entry is visible

        # C. SLEEPING STATE (GRAY)
        else:
            regime_color = "gray"
            display_bias = "HOLD"
            # Targets remain 0.0, but entry is visible

        # ------------------------------------------------------------------
        # 6. EXPLAINABILITY
        # ------------------------------------------------------------------
        drivers = []
        if abs(whale_z) > 1.2:
            drivers.append({"feature": "Volume", "desc": "Whale Activity", "importance": 90})
        if abs(event_alpha) > 1.0:
            drivers.append({"feature": "Event", "desc": "Liquidity Trap", "importance": 85})
        if regime_label == "TRENDING" and abs(trend_alpha) > 1.0:
            drivers.append({"feature": "Trend", "desc": "Market Structure", "importance": 80})

        narrative = self._build_narrative(display_bias, score, regime_label)

        # Risk Reward Calc
        rr_ratio = 0.0
        if stop != 0:
            risk = abs(entry - stop)
            reward = abs(t1 - entry)
            if risk > 0:
                rr_ratio = round(reward / risk, 2)

        return SimpleNamespace(
            bias=display_bias,
            score=int(score),
            entry=entry,  # FIX: Always returns current price
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=rr_ratio,
            expected_duration=expected_duration,
            regime=regime_label,
            regime_color=regime_color,
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers,
            narrative=narrative,
            flow_score=0.5
        )

    def _build_narrative(self, bias, score, regime):
        if bias == "HOLD":
            return f"Neutral market ({regime.lower()}). Waiting for volume."
        if bias == "WATCH":
            return f"Momentum building ({score}%). Waiting for trigger."

        strength = "Strong" if score > 80 else "Moderate"
        return f"{strength} Signal detected in {regime.lower()} conditions."

    def _neutral_result(self, price, reason):
        # FIX: entry=price here too
        return SimpleNamespace(
            bias="HOLD", score=50, entry=price, stop=0.0, target1=0.0, target2=0.0,
            target3=0.0, rr_ratio=0, expected_duration="--", regime="WAIT",
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[],
            narrative=reason, flow_score=0.5
        )