# core/quant/crypto_engine.py
from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


def cap(x, limit=3.0):
    return max(-limit, min(limit, x))


class CryptoQuantEngine:
    """
    REELIOO QUANT PHYSICS ENGINE (v9.4 – Sniper Lane Model)

    The "Code 3" Hybrid:
    - Structure: Lane Model (Attack/Engage) for rich UI feedback.
    - Safety: Strict "Code 1" Gates (0.6 Vol / 0.04 ATR) to prevent loss.
    - Logic: Hard Gating. If conditions are bad, Lane = STAND DOWN.
    - Fix: Full API compatibility (target3, regime, lifecycle).
    """

    def __init__(self):
        self.SIGMOID_K = 0.45

        # --- THRESHOLDS (TUNED FOR PRECISION) ---
        self.ATTACK_THRESH = 75  # 🟢 Sniper Entry (Full Size)
        self.ENGAGE_THRESH = 65  # 🟡 Confirmation Entry (Reduced Size)
        self.PREPARE_THRESH = 50  # 🟠 Watch Mode

        # --- STRICT SAFETY GATES (FROM CODE 1) ---
        self.MIN_VOLUME_RATIO = 0.6  # High standard for liquidity
        self.MAX_ATR_PCT = 0.04  # Strict chop/wick filter

        log.info("🚀 QuantPhysicsEngine v9.4 (Sniper Lane) Online")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        try:
            df = generate_features(df)
            if df.empty: return self._neutral_result(0.0, "No data")

            # Critical: Capture Price
            if "live_close" in df.columns:
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            last = df.iloc[-1]

            # Physics
            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            df['force'] = mass * velocity
            df['friction_coeff'] = df.get('trades', 1) / (mass + 1)
        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # --- AXIS 1: DIRECTIONAL PRESSURE ---
        trend_alpha = cap(last.get("ema_diff", 0) * 100) * 1.5
        whale_z = cap(float(last.get("whale_z", 0)))
        reversion_alpha = -cap(last.get("vwap_dist", 0) * 100) * 1.2

        kinetic = cap(float(last.get("kinetic_energy", 0)))
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        compression = float(last.get("volatility_compression", 1.0))

        physics_alpha = (kinetic * 0.8) + shock
        is_spring_loaded = (compression < 0.6 and abs(whale_z) > 0.8)
        if is_spring_loaded: physics_alpha += (3.0 * np.sign(trend_alpha + shock))

        raw_score = self._sigmoid(trend_alpha + whale_z + reversion_alpha + physics_alpha)

        # --- AXIS 2: HARD GATING (The Sniper Logic) ---
        # We calculate the condition, but we do NOT apply a soft penalty.
        # If the gate fails, we force the lane to "STAND DOWN".

        gate_status = "OPEN"
        gate_reason = ""

        # 1. Friction Gate
        avg_friction = df['friction_coeff'].rolling(20).mean().iloc[-1]
        if last['friction_coeff'] > avg_friction * 1.8:
            gate_status = "CLOSED"
            gate_reason = "High Friction"

        # 2. Liquidity Gate (Strict 0.6 Ratio)
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < (avg_vol * self.MIN_VOLUME_RATIO):
            gate_status = "CLOSED"
            gate_reason = "Low Liquidity"

        # 3. Volatility Gate (Strict 4%)
        if float(last.get("atr_pct", 0)) > self.MAX_ATR_PCT:
            gate_status = "CLOSED"
            gate_reason = "Max Volatility Exceeded"

        # 4. Risk Protocol (Hard Kill)
        kill_switch = False
        if abs(shock) > 2.8:
            kill_switch = True
            gate_status = "CLOSED"
            gate_reason = "Black Swan Event"

        # --- DECISION LOGIC (Lanes with Hard Gates) ---
        lane = "⚫ STAND DOWN"
        bias = "HOLD"

        # If gates are closed, we force score to 50 regardless of alpha
        if gate_status == "CLOSED":
            display_score = 50
        else:
            display_score = int(50 + (raw_score - 50) * 0.95)  # Slight smoothing

            if display_score >= self.ATTACK_THRESH:
                lane = "🟢 ATTACK"
                bias = "LONG"
            elif display_score <= (100 - self.ATTACK_THRESH):
                lane = "🟢 ATTACK"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= self.ENGAGE_THRESH:
                lane = "🟡 ENGAGE"
                bias = "LONG"
            elif display_score <= (100 - self.ENGAGE_THRESH):
                lane = "🟡 ENGAGE"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= self.PREPARE_THRESH:
                lane = "🟠 PREPARE"
                bias = "WATCH"
            elif display_score <= (100 - self.PREPARE_THRESH):
                lane = "🟠 PREPARE"
                bias = "WATCH"
                display_score = 100 - display_score
            else:
                lane = "⚫ STAND DOWN"
                bias = "HOLD"
                display_score = 50

        # --- OUTPUT CONSTRUCTION ---
        entry = stop = t1 = t2 = t3 = 0.0
        if bias in ["LONG", "SHORT"]:
            entry = price
            atr = float(last.get("atr_14", price * 0.01))
            direction = 1 if bias == "LONG" else -1

            # Precision Sizing
            mult = 2.0 if lane == "🟢 ATTACK" else 1.5

            stop = price - direction * (atr * 1.5)
            t1 = price + direction * (atr * mult)
            t2 = price + direction * (atr * mult * 2.0)
            t3 = price + direction * (atr * mult * 3.5)

        drivers = []
        if display_score >= 55:
            if is_spring_loaded: drivers.append({"desc": "Squeeze Setup", "importance": 95})
            if abs(kinetic) > 1.2: drivers.append({"desc": "Surge Momentum", "importance": 85})
            if gate_status == "OPEN": drivers.append({"desc": "Clean Traffic", "importance": 80})
            if gate_status == "CLOSED": drivers.append({"desc": f"Gate: {gate_reason}", "importance": 100})

        narrative = self._build_narrative(lane, display_score, is_spring_loaded, kill_switch, gate_reason)

        # Retail Friendly Regime Logic
        regime_label = "SURGE" if abs(physics_alpha) > 1.5 else "TREND"

        return SimpleNamespace(
            bias=bias,
            lane=lane,
            score=display_score,
            price=price,
            entry=entry,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),  # FIXED
            rr_ratio=2.0 if entry > 0 else 0.0,
            expected_duration="4h",

            regime=regime_label,  # FIXED
            regime_color="green" if lane == "🟢 ATTACK" else "yellow" if lane == "🟡 ENGAGE" else "gray",

            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers[:3],
            narrative=narrative,

            lifecycle="CONFIRMED" if entry > 0 else "EMERGING" if bias == "WATCH" else "WAITING",
            flow_score=0.5
        )

    def _build_narrative(self, lane, score, is_spring, kill_switch, gate_reason):
        if kill_switch: return "🛡️ RISK PROTOCOL: Black Swan protection active."
        if gate_reason: return f"⚠️ {gate_reason}. Trade blocked for safety."
        if lane == "🟢 ATTACK": return "Institutional Force Confirmed. Clear Path."
        if lane == "🟡 ENGAGE": return f"Force present ({score}%). Valid Entry."
        if lane == "🟠 PREPARE": return "Energy building. Awaiting trigger."
        if is_spring: return "⚡ SQUEEZE DETECTED: Volatility Compression."
        return "Market idle. Monitoring for force vectors."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ STAND DOWN", score=50, price=price, entry=0.0,
            stop=0.0, target1=0.0, target2=0.0, target3=0.0,  # FIXED
            rr_ratio=0, expected_duration="--",
            regime="SCANNING", regime_color="gray",
            whale_zscore=0, whale_label="Normal",
            top_features=[], narrative=reason,
            lifecycle="WAITING", flow_score=0.5
        )