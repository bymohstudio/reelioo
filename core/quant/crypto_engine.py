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
    REELIOO QUANT PHYSICS ENGINE (v8.6 – Retail Polished)

    - Renamed "Kill Switch" -> "Risk Protocol"
    - Ensures Market Price is always visible
    - Full Institutional Gatekeeping layers active
    """

    def __init__(self):
        self.SIGMOID_K = 0.45
        self.CONFIRMATION_THRESH = 70

        # --- INSTITUTIONAL CONFIG ---
        self.MIN_VOLUME_RATIO = 0.6  # Liquidity check
        self.MAX_ATR_PCT = 0.04  # Volatility guard

        log.info("🚀 QuantPhysicsEngine v8.6 (Retail Polished) Initialized")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        try:
            # 1. ENRICH DATA
            df = generate_features(df)
            if df.empty: return self._neutral_result(0.0, "No Data")

            # --- CRITICAL: CAPTURE MARKET PRICE IMMEDIATELY ---
            if "live_close" in df.columns:
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            # 2. PHYSICS CALCULATIONS
            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            trades = df.get('trades', 1)

            df['force'] = mass * velocity
            df['friction_coeff'] = trades / (mass + 1)

            last = df.iloc[-1]
        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # ==================================================================
        # PHASE 1: CORE PHYSICS (The Alpha)
        # ==================================================================

        # Stagnation Check (Friction)
        avg_friction = df['friction_coeff'].rolling(20).mean().iloc[-1]
        current_friction = df['friction_coeff'].iloc[-1]
        # High Friction + Low Motion = Trap
        is_stagnant = (current_friction > (avg_friction * 1.5) and abs(last.get('ret_1', 0)) < 0.001)

        # Vector Scoring
        trend_alpha = cap(last.get("ema_diff", 0) * 100) * 1.5
        whale_z = cap(float(last.get("whale_z", 0)))
        reversion_alpha = -cap(last.get("vwap_dist", 0) * 100) * 1.2

        # Physics Vectors
        kinetic = cap(float(last.get("kinetic_energy", 0)))
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        compression = float(last.get("volatility_compression", 1.0))

        physics_alpha = (kinetic * 0.8) + shock
        is_spring_loaded = (compression < 0.6 and abs(whale_z) > 0.8)
        if is_spring_loaded:
            physics_alpha += (3.0 * np.sign(trend_alpha + shock))

        # ==================================================================
        # PHASE 2: INSTITUTIONAL GATEKEEPERS (The Protection)
        # ==================================================================

        # Raw Probability
        raw_alpha = trend_alpha + whale_z + reversion_alpha + physics_alpha
        final_probability = self._sigmoid(raw_alpha)

        # 1. Session Awareness
        session_mod = 0
        try:
            hour = last.name.hour
            if 0 <= hour < 8:
                session_mod = -5  # Asia Discount
            elif 13 <= hour < 16:
                session_mod = +5  # NY/London Premium
        except:
            pass

        adjusted_score = final_probability + session_mod

        # 2. Execution Gates
        execution_gate = True
        gate_reason = ""

        # Liquidity Check
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < (avg_vol * self.MIN_VOLUME_RATIO):
            execution_gate = False
            gate_reason = "Low Liquidity"

        # Volatility Shock Check
        if float(last.get("atr_pct", 0)) > self.MAX_ATR_PCT:
            execution_gate = False
            gate_reason = "Max Volatility Exceeded"

        # 3. Kill Switch -> Risk Protocol
        kill_switch = False
        if abs(shock) > 2.8:
            kill_switch = True
            gate_reason = "Extreme Volatility"

        # ==================================================================
        # PHASE 3: FINAL DECISION
        # ==================================================================

        # Force Neutrality if Gates Fail
        if is_stagnant or not execution_gate or kill_switch:
            adjusted_score = 50

        # Determine Bias
        score = int(adjusted_score)
        bias = "HOLD"

        if score >= 70:
            bias = "LONG"
        elif score <= 30:
            bias = "SHORT"
            score = 100 - score
        elif 60 <= score < 70:
            bias = "WATCH"
        elif 30 < score <= 40:
            bias = "WATCH"
            score = 100 - score
        else:
            bias = "HOLD"
            score = 50

        # Score Smoothing (UI Stability)
        score = 50 + (score - 50) * 0.95

        # ==================================================================
        # PHASE 4: OUTPUT CONSTRUCTION
        # ==================================================================

        # Lifecycle State
        lifecycle = "WAITING"
        if bias == "WATCH": lifecycle = "EMERGING"
        if bias in ["LONG", "SHORT"]: lifecycle = "CONFIRMED"
        if kill_switch: lifecycle = "SHIELDED"

        # Trade Levels (Hidden unless Confirmed)
        entry = 0.0
        stop = 0.0
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        expected_duration = "--"
        rr_ratio = 0.0

        if bias in ["LONG", "SHORT"]:
            entry = price
            atr = float(last.get("atr_14", price * 0.01))
            direction = 1 if bias == "LONG" else -1

            # Physics Extension
            extension = 1.0 + (abs(physics_alpha) * 0.2)

            stop = price - (direction * atr * 1.5)
            t1 = price + (direction * atr * 2.0 * extension)
            t2 = price + (direction * atr * 3.5 * extension)
            t3 = price + (direction * atr * 5.0 * extension)

            risk = abs(entry - stop)
            if risk > 0:
                rr_ratio = round(abs(t1 - entry) / risk, 2)
            expected_duration = "4h"

        # Explainability (RETAIL FRIENDLY TERMS)
        drivers = []
        if score >= 60 or is_stagnant or kill_switch:
            if kill_switch:
                # RENAME: Kill Switch -> Risk Protocol
                drivers.append({"desc": f"RISK PROTOCOL: {gate_reason}", "importance": 100})
            elif is_stagnant:
                drivers.append({"desc": "Retail Trap (Choppy)", "importance": 99})
            else:
                # Retail Friendly Mappings
                if int(last.get("liq_sweep", 0)) != 0: drivers.append({"desc": "Stop Hunt", "importance": 95})
                if is_spring_loaded:
                    drivers.append({"desc": "Squeeze Setup", "importance": 98})
                elif abs(kinetic) > 1.5:
                    drivers.append({"desc": "Momentum", "importance": 92})

                if abs(whale_z) > 1.2: drivers.append({"desc": "Smart Money", "importance": 85})
                if abs(trend_alpha) > 1.2: drivers.append({"desc": "Trend", "importance": 75})

        # Limit to top 3
        drivers = sorted(drivers, key=lambda x: x.get('importance', 0), reverse=True)[:3]

        narrative = self._build_narrative(bias, score, is_stagnant, is_spring_loaded, kill_switch, gate_reason)

        return SimpleNamespace(
            bias=bias,
            score=int(score),
            price=price,
            entry=entry,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=rr_ratio,
            expected_duration=expected_duration,
            regime="IMPULSE" if abs(physics_alpha) > 1.5 else "FLOW",
            regime_color="green" if bias == "LONG" else "red" if bias == "SHORT" else "yellow" if bias == "WATCH" else "gray",
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers,
            narrative=narrative,
            lifecycle=lifecycle,
            flow_score=0.5
        )

    def _build_narrative(self, bias, score, is_stagnant, is_spring, kill_switch, gate_reason):
        if kill_switch:
            return f"🛡️ RISK PROTOCOL ACTIVE: {gate_reason}. Capital Preserved."
        if gate_reason:
            return f"⚠️ Gate Closed: {gate_reason}."
        if is_stagnant:
            return "⚠️ Retail Trap Detected. Market is chopping."
        if bias == "HOLD":
            return "System Idle. No Edge."
        if bias == "WATCH":
            return f"Momentum building ({score}%)."
        if is_spring:
            return "⚡ SQUEEZE DETECTED: Explosive move imminent."

        return f"Confirmed institutional force detected."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", score=50,
            price=price, entry=0.0,
            stop=0.0, target1=0.0, target2=0.0,
            target3=0.0, rr_ratio=0, expected_duration="--", regime="WAIT",
            regime_color="gray", whale_zscore=0, whale_label="Normal", top_features=[],
            narrative=reason, flow_score=0.5
        )