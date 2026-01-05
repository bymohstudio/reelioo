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
    REELIOO QUANT PHYSICS ENGINE (v5.4 – Kinetic Integration)

    - Integrated New Physics Vectors:
      1. Kinetic Energy (Mass * Velocity): Validates move strength.
      2. Momentum Shock (Jerk): Detects instant acceleration.
      3. Volatility Compression (Spring): Identifies explosive breakouts.
    """

    def __init__(self):
        self.SIGMOID_K = 0.45
        self.CONFIRMATION_THRESH = 70
        log.info("🚀 QuantPhysicsEngine v5.4 (Kinetic) Initialized")

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
        # 2. STANDARD VECTORS (Layer 1)
        # ------------------------------------------------------------------
        # Trend
        ema_diff = cap(last.get("ema_diff", 0) * 100)
        rsi_z = cap((last.get("rsi_14", 50) - 50) / 15)
        trend_alpha = (ema_diff * 1.5 + rsi_z) * (1.0 if regime_label == "TRENDING" else 0.5)
        trend_alpha = cap(trend_alpha)

        # Whale (Volume Z-Score)
        whale_z = cap(float(last.get("whale_z", 0)))

        # Reversion
        vwap_z = cap(last.get("vwap_dist", 0) * 100)
        reversion_alpha = -vwap_z * (2.0 if regime_label == "CHOPPY" else 1.2)
        reversion_alpha = cap(reversion_alpha)

        # Events (Liquidity Sweeps)
        event_alpha = 0.0
        if int(last.get("liq_sweep", 0)) == 1:
            event_alpha += 2.0
        elif int(last.get("liq_sweep", 0)) == -1:
            event_alpha -= 2.0

        # CVD Divergence
        if int(last.get("cvd_divergence", 0)) == 1:
            event_alpha += 1.5
        elif int(last.get("cvd_divergence", 0)) == -1:
            event_alpha -= 1.5
        event_alpha = cap(event_alpha)

        # ------------------------------------------------------------------
        # 3. PHYSICS VECTORS (Layer 2 - The Profit Multipliers)
        # ------------------------------------------------------------------
        # Extract new features
        kinetic = cap(float(last.get("kinetic_energy", 0)))
        shock = cap(float(last.get("momentum_shock", 0)) * 5)  # Scale shock up as values are small
        compression = float(last.get("volatility_compression", 1.0))

        physics_alpha = 0.0

        # A. KINETIC ENERGY (Validation)
        # If Kinetic Energy is high (>1.5), the move has "Mass". Hard to fake.
        # We allow Kinetic Energy to boost the existing trend direction.
        if abs(kinetic) > 1.2:
            physics_alpha += kinetic * 0.8

        # B. MOMENTUM SHOCK (Leading Indicator)
        # Detects immediate acceleration/jerk. Great for early entries.
        physics_alpha += shock

        # C. COMPRESSION (The Spring Setup - MOST PROFITABLE)
        # If Volatility is compressed (<0.6) AND Whales are active (>0.8)...
        # This is a "Spring Loaded" setup. Massive boost to probability.
        is_spring_loaded = False
        if compression < 0.6 and abs(whale_z) > 0.8:
            is_spring_loaded = True
            # Determine direction based on existing flow
            direction = 1.0 if (trend_alpha + shock) > 0 else -1.0
            physics_alpha += (3.0 * direction)  # Huge Alpha Boost

        physics_alpha = cap(physics_alpha)

        # ------------------------------------------------------------------
        # 4. TOTAL PROBABILITY
        # ------------------------------------------------------------------
        raw_alpha = trend_alpha + whale_z + reversion_alpha + event_alpha + physics_alpha
        final_probability = self._sigmoid(raw_alpha)

        # Stability Dampener (Fakeout Filter)
        # If volatility is expanding rapidly but there is NO Kinetic Mass (Low Volume),
        # it's a fakeout (Pump & Dump). Reduce probability.
        vol_slope = float(last.get("volatility_slope", 0))
        if vol_slope > 0.25 and abs(kinetic) < 0.5:
            # Drag score back to 50
            final_probability = 50 + (final_probability - 50) * 0.6

        # ------------------------------------------------------------------
        # 5. INITIAL BIAS & SCORE
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

        # Score Compression (Make 90+ harder to get)
        score = 50 + (score - 50) * 0.90

        # ------------------------------------------------------------------
        # 6. VISUAL STATE & LOGIC
        # ------------------------------------------------------------------
        thresh = self.CONFIRMATION_THRESH - (5 if trade_style == "SCALP" else 0)

        regime_color = "gray"
        display_bias = bias

        # UI Fix: Entry always shows price
        entry = price
        stop, t1, t2, t3 = 0.0, 0.0, 0.0, 0.0
        expected_duration = "--"

        # A. CONFIRMED TRADE
        if score >= thresh:
            regime_color = "green" if bias == "LONG" else "red"
            display_bias = bias

            atr = float(last.get("atr_14", price * 0.01))
            stop_mult, tgt_mult = (1.0, 1.5) if trade_style == "SCALP" else (1.5, 3.0)

            # If "Spring Loaded", extend targets because expansion will be large
            if is_spring_loaded:
                tgt_mult += 1.5
                expected_duration = "Rapid Expansion"
            else:
                expected_duration = "4h - 24h"

            direction = 1 if bias == "LONG" else -1
            stop = price - direction * atr * stop_mult
            t1 = price + direction * atr * tgt_mult
            t2 = t1 + (abs(t1 - price) * 0.5) * direction
            t3 = t1 + abs(t1 - price) * direction

        # B. WATCHING STATE
        elif score >= 60:
            regime_color = "yellow"
            display_bias = "WATCH"

        # C. HOLD STATE
        else:
            regime_color = "gray"
            display_bias = "HOLD"

        # ------------------------------------------------------------------
        # 7. EXPLAINABILITY (Updated for Physics)
        # ------------------------------------------------------------------
        drivers = []

        # Priority 1: Physics Drivers
        if is_spring_loaded:
            drivers.append({"feature": "Physics", "desc": "Spring Loaded (Compression)", "importance": 98})
        elif abs(kinetic) > 1.5:
            drivers.append({"feature": "Physics", "desc": "High Kinetic Energy", "importance": 92})
        elif abs(shock) > 1.5:
            drivers.append({"feature": "Physics", "desc": "Momentum Shock", "importance": 88})

        # Priority 2: Standard Drivers
        if abs(whale_z) > 1.2:
            drivers.append({"feature": "Volume", "desc": "Whale Activity", "importance": 85})
        if abs(event_alpha) > 1.0:
            drivers.append({"feature": "Event", "desc": "Liquidity Trap", "importance": 80})

        # Fallback
        if not drivers and regime_label == "TRENDING":
            drivers.append({"feature": "Trend", "desc": "Market Structure", "importance": 75})

        narrative = self._build_narrative(display_bias, score, regime_label, is_spring_loaded)

        # Risk Reward
        rr_ratio = 0.0
        if stop != 0:
            risk = abs(entry - stop)
            reward = abs(t1 - entry)
            if risk > 0:
                rr_ratio = round(reward / risk, 2)

        return SimpleNamespace(
            bias=display_bias,
            score=int(score),
            entry=entry,
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
            top_features=drivers[:3],  # Top 3 only
            narrative=narrative,
            flow_score=0.5
        )

    def _build_narrative(self, bias, score, regime, is_spring=False):
        if bias == "HOLD":
            return f"Neutral market ({regime.lower()}). Waiting for energy."
        if bias == "WATCH":
            return f"Kinetic energy building ({score}%). Waiting for trigger."

        if is_spring:
            return "⚠️ SPRING LOADED: Volatility compression detected. Explosive move imminent."

        strength = "Strong" if score > 80 else "Moderate"
        return f"{strength} Signal confirmed by physics engine."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", score=50, entry=price, stop=0.0, target1=0.0, target2=0.0,
            target3=0.0, rr_ratio=0, expected_duration="--", regime="WAIT",
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[],
            narrative=reason, flow_score=0.5
        )