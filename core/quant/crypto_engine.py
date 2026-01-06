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
    REELIOO QUANT PHYSICS ENGINE (v9.5 – Trend-Vector Logic)

    - FIX: Removed 'Shock' from Alpha (stops chasing wicks).
    - FIX: Added Vector Alignment (Trend + Physics must agree).
    - LOGIC: Trend Following + Kinetic Energy Confirmation.
    """

    def __init__(self):
        self.SIGMOID_K = 0.5  # Slightly steeper curve for decisiveness

        # --- THRESHOLDS ---
        self.ATTACK_THRESH = 75  # 🟢 Sniper Entry
        self.ENGAGE_THRESH = 65  # 🟡 Confirmation Entry
        self.PREPARE_THRESH = 50  # 🟠 Watch Mode

        # --- GATES ---
        self.MIN_VOLUME_RATIO = 0.6
        self.MAX_ATR_PCT = 0.04

        log.info("🚀 QuantPhysicsEngine v9.5 (Trend-Vector) Online")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        try:
            # 1. Generate Base Features
            df = generate_features(df)
            if df.empty: return self._neutral_result(0.0, "No data")

            # 2. Capture Price
            if "live_close" in df.columns:
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            # 3. Physics Calculations
            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            df['force'] = mass * velocity
            df['friction_coeff'] = df.get('trades', 1) / (mass + 1)

            # 4. Capture Last Row (After calculations)
            last = df.iloc[-1]

        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # ==========================================
        # PHASE 1: VECTOR ANALYSIS (The Fix)
        # ==========================================

        # 1. Trend Vector (Direction)
        # Weight: High. We want to follow the river, not swim upstream.
        trend_alpha = cap(last.get("ema_diff", 0) * 100) * 2.0

        # 2. Whale Vector (Confirmation)
        # Weight: Medium. Big money must support the move.
        whale_z = cap(float(last.get("whale_z", 0))) * 1.0

        # 3. Physics Vector (Energy)
        # Weight: High. Is there actual velocity behind this?
        # NOTE: We removed 'shock' from here. Only pure Kinetic Energy.
        kinetic = cap(float(last.get("kinetic_energy", 0))) * 1.5

        # 4. Alignment Check (Crucial Fix)
        # If Trend is UP but Physics is DOWN, we kill the signal.
        vector_mismatch = False
        if np.sign(trend_alpha) != np.sign(kinetic) and abs(trend_alpha) > 0.5 and abs(kinetic) > 0.5:
            vector_mismatch = True

        # Calculate Score
        # We removed 'reversion_alpha' to stop betting against trends.
        raw_alpha = trend_alpha + whale_z + kinetic
        raw_score = self._sigmoid(raw_alpha)

        # ==========================================
        # PHASE 2: SAFETY GATES
        # ==========================================
        gate_status = "OPEN"
        gate_reason = ""

        # 1. Friction Gate (Sticky Market)
        avg_friction = df['friction_coeff'].rolling(20).mean().iloc[-1]
        if pd.isna(avg_friction): avg_friction = 0
        if last['friction_coeff'] > (avg_friction * 1.8) and avg_friction > 0:
            gate_status = "CLOSED"
            gate_reason = "High Friction"

        # 2. Liquidity Gate (No Volume = No Trade)
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < (avg_vol * self.MIN_VOLUME_RATIO):
            gate_status = "CLOSED"
            gate_reason = "Low Liquidity"

        # 3. Volatility Gate (Anti-Wick)
        if float(last.get("atr_pct", 0)) > self.MAX_ATR_PCT:
            gate_status = "CLOSED"
            gate_reason = "Max Volatility"

        # 4. Shock Gate (The New Home for Shock)
        # We only use Shock to BLOCK trades, never to create them.
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        kill_switch = False
        if abs(shock) > 2.8:
            kill_switch = True
            gate_status = "CLOSED"
            gate_reason = "Black Swan Event"

        # 5. Vector Gate (The Mismatch Fix)
        if vector_mismatch:
            gate_status = "CLOSED"
            gate_reason = "Vector Mismatch"

        # ==========================================
        # PHASE 3: LANE LOGIC
        # ==========================================
        lane = "⚫ STAND DOWN"
        bias = "HOLD"

        if gate_status == "CLOSED":
            display_score = 50
        else:
            display_score = int(50 + (raw_score - 50) * 0.95)

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

        # ==========================================
        # PHASE 4: OUTPUT
        # ==========================================
        entry = stop = t1 = t2 = t3 = 0.0
        if bias in ["LONG", "SHORT"]:
            entry = price
            atr = float(last.get("atr_14", price * 0.01))
            direction = 1 if bias == "LONG" else -1

            # Physics-Adjusted Targets
            # If kinetic energy is high, we extend targets
            extension = 1.0 + (abs(kinetic) * 0.3)

            stop = price - direction * (atr * 1.5)
            t1 = price + direction * (atr * 2.0)
            t2 = price + direction * (atr * 3.5 * extension)
            t3 = price + direction * (atr * 5.0 * extension)

        # Explainability Drivers
        drivers = []
        if display_score >= 55:
            if abs(kinetic) > 1.0: drivers.append({"desc": "Kinetic Drive", "importance": 90})
            if abs(trend_alpha) > 1.0: drivers.append({"desc": "Trend Alignment", "importance": 85})
            if abs(whale_z) > 1.0: drivers.append({"desc": "Whale Support", "importance": 80})
            if gate_status == "CLOSED": drivers.append({"desc": f"Blocked: {gate_reason}", "importance": 100})

        narrative = self._build_narrative(lane, display_score, gate_reason)

        # Retail Friendly Labels
        regime_label = "SURGE" if abs(kinetic) > 1.2 else "FLOW"

        return SimpleNamespace(
            bias=bias,
            lane=lane,
            score=display_score,
            price=price,
            entry=entry,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=2.0 if entry > 0 else 0.0,
            expected_duration="4h",
            regime=regime_label,
            regime_color="green" if lane == "🟢 ATTACK" else "yellow" if lane == "🟡 ENGAGE" else "gray",
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers[:3],
            narrative=narrative,
            lifecycle="CONFIRMED" if entry > 0 else "EMERGING" if bias == "WATCH" else "WAITING",
            flow_score=0.5
        )

    def _build_narrative(self, lane, score, gate_reason):
        if gate_reason: return f"⚠️ {gate_reason}. Signal blocked."
        if lane == "🟢 ATTACK": return "Full Alignment: Trend + Physics + Whales."
        if lane == "🟡 ENGAGE": return f"Trend Confirmed ({score}%). Monitor Friction."
        if lane == "🟠 PREPARE": return "Energy building. Awaiting momentum."
        return "Market idle. Scanning for institutional vectors."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ STAND DOWN", score=50, price=price, entry=0.0,
            stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0, expected_duration="--",
            regime="SCANNING", regime_color="gray",
            whale_zscore=0, whale_label="Normal",
            top_features=[], narrative=reason,
            lifecycle="WAITING", flow_score=0.5
        )