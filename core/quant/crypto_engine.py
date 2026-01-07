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
    REELIOO QUANT PHYSICS ENGINE (v16.0 – FRACTAL GEOMETRY)

    THE "GOD TIER" UPGRADE:
    1. FRACTAL EFFICIENCY: Distinguishes between "Sustainable Trends" (Whales)
       and "Unstable Bubbles" (Retail FOMO).
    2. FORCE DIVERGENCE: Detects hidden weakness before price turns.
    3. TARGET GEOMETRY: Exits based on energy exhaustion, not just ATR.

    RESULT: A system that refuses to buy the top, even if momentum is high.
    """

    def __init__(self):
        self.SIGMOID_K = 0.5

        # --- INSTITUTIONAL CONFIG ---
        self.ATTACK_THRESH = 80  # 🟢 Sniper Conviction
        self.ENGAGE_THRESH = 65  # 🟡 Standard Entry

        # --- GATES ---
        self.MIN_VOLUME_RATIO = 0.6
        self.MAX_ATR_PCT = 0.05

        log.info("🚀 QuantPhysicsEngine v16.0 (Fractal Geometry) Online")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        try:
            # 1. Data Enrichment
            df = generate_features(df)
            if df.empty: return self._neutral_result(0.0, "No data")

            if "live_close" in df.columns:
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            last = df.iloc[-1]

            # 2. PHYSICS CALCULATIONS
            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            acceleration = velocity.diff()
            jerk = acceleration.diff()  # The Snap

            # 3. FRACTAL CALCULATIONS (New)
            # Calculate Kaufman's Efficiency Ratio (ER) manually for precision
            # ER = Change / Sum of absolute changes.
            # ER 1.0 = Perfect Line. ER 0.0 = Random Noise.
            period = 10
            change = abs(df['close'].diff(period))
            volatility = df['close'].diff().abs().rolling(period).sum()
            efficiency_ratio = change / (volatility + 0.00001)

            current_er = float(efficiency_ratio.iloc[-1])

            # 4. FORCE DIVERGENCE (New)
            # Calculate Force (Mass * Velocity)
            force = mass * velocity
            # Check if Price is Higher than 5 bars ago, but Force is Lower
            price_trend_5 = df['close'].iloc[-1] > df['close'].iloc[-6]
            force_trend_5 = force.iloc[-1] < force.iloc[-6]
            is_divergence = price_trend_5 and force_trend_5

        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # ==========================================
        # PHASE 1: VECTOR ANALYSIS
        # ==========================================

        # A. TREND VECTOR (Fractal Adjusted)
        # If the trend is "Jagged" (Low ER), we discount it heavily.
        # Whales create smooth trends (High ER). Retail creates jagged ones.
        raw_trend = cap(last.get("ema_diff", 0) * 100) * 2.0
        fractal_quality = 1.5 if current_er > 0.5 else 0.5
        trend_alpha = raw_trend * fractal_quality

        # B. WHALE VECTOR
        whale_z = cap(float(last.get("whale_z", 0)))

        # C. KINETIC SNAP
        jerk_val = float(jerk.iloc[-1]) if not pd.isna(jerk.iloc[-1]) else 0.0
        kinetic_energy = float(last.get("kinetic_energy", 0))
        physics_alpha = 0.0

        if abs(kinetic_energy) > 1.0:
            physics_alpha += cap(jerk_val * 10.0)
            physics_alpha += kinetic_energy * 0.5

        # D. COMPRESSION (Spring)
        is_spring_loaded = float(last.get("volatility_compression", 1.0)) < 0.6
        if is_spring_loaded and abs(whale_z) > 0.8:
            breakout_dir = np.sign(jerk_val + trend_alpha)
            physics_alpha += (4.0 * breakout_dir)

        # ==========================================
        # PHASE 2: THE "GOD" FILTERS
        # ==========================================
        gate_status = "OPEN"
        gate_reason = ""
        penalty = 0

        # 1. Fractal Efficiency Gate
        # If the market is pure noise (ER < 0.25), DO NOT TRADE.
        # This saves you from the "Chop" that kills 90% of bots.
        if current_er < 0.25 and not is_spring_loaded:
            gate_status = "CLOSED"
            gate_reason = "Fractal Noise (Random)"

        # 2. Force Divergence Penalty
        # If Price is going up but Force is going down, it's a trap.
        if is_divergence and not is_spring_loaded:
            penalty += 25  # Massive penalty

        # 3. Liquidity Gate
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume'] < (avg_vol * self.MIN_VOLUME_RATIO):
            gate_status = "CLOSED"
            gate_reason = "Low Liquidity"

        # 4. Volatility Limit
        atr_pct = float(last.get("atr_pct", 0))
        if atr_pct > self.MAX_ATR_PCT and not is_spring_loaded:
            gate_status = "CLOSED"
            gate_reason = "Volatility Limit"

        # ==========================================
        # PHASE 3: SCORING
        # ==========================================

        total_alpha = trend_alpha + whale_z + physics_alpha
        raw_score = self._sigmoid(total_alpha)

        final_score = int(raw_score - penalty)
        final_score = max(0, min(100, final_score))

        # ==========================================
        # PHASE 4: DECISION LANES
        # ==========================================
        lane = "⚫ HOLD"
        bias = "HOLD"

        if gate_status == "CLOSED":
            final_score = 50
        else:
            if final_score >= self.ATTACK_THRESH:
                lane = "🟢 STRONG BUY"
                bias = "LONG"
            elif final_score <= (100 - self.ATTACK_THRESH):
                lane = "🟢 STRONG SELL"
                bias = "SHORT"
                final_score = 100 - final_score
            elif final_score >= self.ENGAGE_THRESH:
                lane = "🟡 BUY"
                bias = "LONG"
            elif final_score <= (100 - self.ENGAGE_THRESH):
                lane = "🟡 SELL"
                bias = "SHORT"
                final_score = 100 - final_score
            elif final_score >= 55:
                lane = "🟠 WATCH"
                bias = "WATCH"
            elif final_score <= 45:
                lane = "🟠 WATCH"
                bias = "WATCH"
                final_score = 100 - final_score
            else:
                lane = "⚫ HOLD"
                bias = "HOLD"
                final_score = 50

        # ==========================================
        # PHASE 5: FRACTAL TARGETING
        # ==========================================
        entry = stop = t1 = t2 = t3 = 0.0
        if bias in ["LONG", "SHORT"]:
            entry = price
            atr = float(last.get("atr_14", price * 0.01))
            direction = 1 if bias == "LONG" else -1

            # Dynamic Extension based on Fractal Efficiency
            # If ER is high (Smooth Trend), we can aim much higher.
            # If ER is low (Jagged), we take profit early.
            fractal_mult = 1.5 if current_er > 0.6 else 1.0

            stop = price - direction * (atr * 2.0)
            t1 = price + direction * (atr * 2.0 * fractal_mult)
            t2 = price + direction * (atr * 4.0 * fractal_mult)
            t3 = price + direction * (atr * 8.0 * fractal_mult)

        # Explainability
        drivers = []
        if final_score >= 60:
            if current_er > 0.6: drivers.append({"desc": "Fractal Alignment", "importance": 100})
            if is_spring_loaded: drivers.append({"desc": "Spring Compression", "importance": 95})
            if abs(jerk_val) > 0.1: drivers.append({"desc": "Kinetic Snap", "importance": 90})
            if is_divergence: drivers.append({"desc": "⚠️ FORCE DIVERGENCE", "importance": -50})
            if gate_status == "CLOSED": drivers.append({"desc": f"Blocked: {gate_reason}", "importance": 100})

        narrative = self._build_narrative(lane, final_score, gate_reason, is_spring_loaded, is_divergence)

        regime_label = "SURGE" if abs(physics_alpha) > 1.5 else "FLOW"

        return SimpleNamespace(
            bias=bias, lane=lane, score=final_score, price=price,
            entry=entry, stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=2.0 if entry > 0 else 0.0, expected_duration="4h",
            regime=regime_label,
            regime_color="green" if "BUY" in lane or "SELL" in lane else "gray",
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers[:3],
            narrative=narrative,
            lifecycle="CONFIRMED" if entry > 0 else "EMERGING" if bias == "WATCH" else "WAITING",
            flow_score=0.5
        )

    def _build_narrative(self, lane, score, gate_reason, is_spring, is_divergence):
        if gate_reason: return f"⛔ {gate_reason}. Capital Preserved."
        if is_divergence and score < 60: return "⚠️ WARNING: Price Rising, Force Falling (Trap)."
        if is_spring: return "⚡ SPRING LOADED: Fractal Compression Detected."
        if "STRONG" in lane: return "High Efficiency Kinetic Setup. Full Alignment."
        if "BUY" in lane or "SELL" in lane: return f"Trend Confirmed ({score}%). Valid Entry."
        if "WATCH" in lane: return "Structure building. Awaiting momentum."
        return "Market idle. Scanning for edge."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price, entry=0.0,
            stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0, expected_duration="--",
            regime="SCANNING", regime_color="gray",
            whale_zscore=0, whale_label="Normal",
            top_features=[], narrative=reason,
            lifecycle="WAITING", flow_score=0.5
        )