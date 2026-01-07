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
    REELIOO QUANT PHYSICS ENGINE (v11.0 – Regime Orchestrator)

    - LAYER 1: Physics (Mass/Velocity) - Unchanged.
    - LAYER 2: Temporal (Stability) - Unchanged.
    - LAYER 3: REGIME ADAPTABILITY (NEW)
      The engine now identifies the asset class (Macro/Beta/Impulse)
      and dynamically loads a physics profile.

      * BTC needs deep stability.
      * PEPE needs fast reaction.
    """

    def __init__(self):
        self.SIGMOID_K = 0.5
        log.info("🚀 QuantPhysicsEngine v11.0 (Regime Orchestrator) Online")

    def _get_asset_profile(self, symbol: str):
        """
        Determines the physics profile based on the asset class.
        In a real hedge fund, this would query a master database.
        Here, we use deterministic mapping.
        """
        s = symbol.upper()

        # PROFILE 1: MACRO LEADERS (Heavy Mass)
        # Needs persistent evidence to turn the ship.
        if any(x in s for x in ['BTC', 'ETH', 'BNB']):
            return {
                'type': 'MACRO',
                'stability_window': 6,  # Look back 6 bars (Slow)
                'min_stability': 0.7,  # 70% of bars must agree
                'vol_gate': 0.04,  # Strict volatility (4%)
                'penalty_mult': 12.0,  # High penalties for deviation
                'attack_thresh': 78  # Harder to trigger ATTACK
            }

        # PROFILE 2: IMPULSE / MEME (Low Mass)
        # Explodes fast. Waiting 6 bars means missing the move.
        elif any(x in s for x in ['PEPE', 'WIF', 'DOGE', 'SHIB', 'BONK', 'FLOKI']):
            return {
                'type': 'IMPULSE',
                'stability_window': 3,  # Look back 3 bars (Fast)
                'min_stability': 0.5,  # Only 50% need to agree (Chaos allowed)
                'vol_gate': 0.08,  # Loose volatility (8% allowed)
                'penalty_mult': 6.0,  # Low penalties (Forgive wicks)
                'attack_thresh': 72  # Easier to trigger ATTACK
            }

        # PROFILE 3: BETA TRENDERS (Medium Mass)
        # The standard "Follower" assets.
        else:
            return {
                'type': 'BETA',
                'stability_window': 5,  # Standard 5 bars
                'min_stability': 0.6,  # 60% agreement
                'vol_gate': 0.06,  # Standard volatility (6%)
                'penalty_mult': 10.0,  # Standard penalties
                'attack_thresh': 75  # Standard threshold
            }

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0

        # 0. DETECT REGIME
        # We do this first to set the rules of engagement.
        symbol_name = str(df.get("symbol", "UNKNOWN"))  # Ensure symbol is passed or inferred
        # Fallback if symbol isn't in dataframe metadata (often passed in controller)
        # For this snippet, we assume caller handles symbol awareness or we default to BETA
        profile = self._get_asset_profile(symbol_name)

        try:
            # 1. Generate Features
            df = generate_features(df)
            if df.empty: return self._neutral_result(0.0, "No data")

            if "live_close" in df.columns:
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            df['force'] = mass * velocity
            df['friction_coeff'] = df.get('trades', 1) / (mass + 1)

            # --- DYNAMIC WINDOW (The Fix) ---
            # We use the profile's specific window size
            window_size = profile['stability_window']
            recent_window = df.iloc[-window_size:]
            last = df.iloc[-1]

        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # ==========================================
        # PHASE 1: PHYSICS & TEMPORAL STABILITY
        # ==========================================

        # 1. Trend Vector
        current_trend = cap(last.get("ema_diff", 0) * 100) * 2.0

        # Stability Check (Dynamic)
        trend_sign = np.sign(current_trend)
        # Calculate how many recent bars match the current trend direction
        agreement_count = sum(np.sign(row) == trend_sign for row in recent_window["ema_diff"])
        stability_ratio = agreement_count / window_size

        # Apply Stability Weight
        trend_alpha = current_trend * stability_ratio

        # 2. Whale Vector
        whale_z = cap(float(last.get("whale_z", 0))) * 1.0

        # 3. Physics Vector
        kinetic = cap(float(last.get("kinetic_energy", 0))) * 1.5

        # 4. Alignment & Dynamic Penalty
        mismatch_penalty = 0
        is_pullback = False

        if np.sign(trend_alpha) != np.sign(kinetic) and abs(trend_alpha) > 0.5:
            is_pullback = True

            atr = float(last.get("atr_14", 1.0))
            current_move_pct = abs(velocity.iloc[-1]) / price
            volatility_ratio = current_move_pct / (atr / price + 0.0001)

            # DYNAMIC MULTIPLIER (Based on Asset Class)
            # Impulse assets get lower penalties for volatility than Macro assets
            mismatch_penalty = profile['penalty_mult'] * max(1.0, min(3.0, volatility_ratio))

        raw_alpha = trend_alpha + whale_z + kinetic
        raw_score = self._sigmoid(raw_alpha)

        # ==========================================
        # PHASE 2: REGIME-AWARE GATES
        # ==========================================
        gate_status = "OPEN"
        gate_reason = ""

        # 1. Temporal Gate (Dynamic Threshold)
        # BTC needs 70% stability. PEPE only needs 50%.
        if stability_ratio < profile['min_stability']:
            gate_status = "CLOSED"
            gate_reason = "Trend Unstable"

        # 2. Friction Gate
        avg_friction = df['friction_coeff'].rolling(20).mean().iloc[-1]
        if pd.isna(avg_friction): avg_friction = 0

        # Impulse assets tolerate higher friction (hype traffic)
        friction_limit = 2.2 if profile['type'] == 'IMPULSE' else 1.8
        if last['friction_coeff'] > (avg_friction * friction_limit) and avg_friction > 0:
            gate_status = "CLOSED"
            gate_reason = "High Friction"

        # 3. Volatility Gate (Dynamic)
        # PEPE allows 8% candles. BTC blocks at 4%.
        if float(last.get("atr_pct", 0)) > profile['vol_gate']:
            gate_status = "CLOSED"
            gate_reason = "Max Volatility"

        # 4. Risk Protocol
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        kill_switch = False
        if abs(shock) > 2.8:
            kill_switch = True
            gate_status = "CLOSED"
            gate_reason = "Black Swan Event"

        # ==========================================
        # PHASE 3: LANE LOGIC
        # ==========================================
        lane = "⚫ STAND DOWN"
        bias = "HOLD"

        if gate_status == "CLOSED":
            display_score = 50
        else:
            base_score = int(50 + (raw_score - 50) * 0.95)
            display_score = int(base_score - mismatch_penalty)
            display_score = max(0, min(100, display_score))

            # Dynamic Thresholds based on Asset Class
            if display_score >= profile['attack_thresh']:
                lane = "🟢 ATTACK"
                bias = "LONG"
            elif display_score <= (100 - profile['attack_thresh']):
                lane = "🟢 ATTACK"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= 65:  # Keep engage standard
                lane = "🟡 ENGAGE"
                bias = "LONG"
            elif display_score <= 35:
                lane = "🟡 ENGAGE"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= 50:
                lane = "🟠 PREPARE"
                bias = "WATCH"
            else:
                lane = "🟠 PREPARE"
                bias = "WATCH"
                display_score = 100 - display_score

            if display_score < 50:  # Safety catch
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

            # IMPULSE assets get wider targets
            # MACRO assets get tighter targets
            extension_mult = 1.5 if profile['type'] == 'IMPULSE' else 1.0
            if not is_pullback:
                extension_mult += (abs(kinetic) * 0.3)

            stop = price - direction * (atr * 1.5)
            t1 = price + direction * (atr * 2.0 * extension_mult)
            t2 = price + direction * (atr * 3.5 * extension_mult)
            t3 = price + direction * (atr * 5.0 * extension_mult)

        drivers = []
        if display_score >= 55:
            drivers.append({"desc": f"Asset Class: {profile['type']}", "importance": 100})
            if stability_ratio >= profile['min_stability']: drivers.append({"desc": "Trend Stable", "importance": 90})
            if is_pullback:
                drivers.append({"desc": "Healthy Pullback", "importance": 85})
            elif abs(kinetic) > 1.0:
                drivers.append({"desc": "Kinetic Drive", "importance": 90})
            if gate_status == "CLOSED": drivers.append({"desc": f"Blocked: {gate_reason}", "importance": 100})

        narrative = self._build_narrative(lane, display_score, gate_reason, is_pullback, profile['type'])
        regime_label = "SURGE" if abs(kinetic) > 1.2 else "FLOW"

        return SimpleNamespace(
            bias=bias, lane=lane, score=display_score, price=price,
            entry=entry, stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=2.0 if entry > 0 else 0.0, expected_duration="4h",
            regime=regime_label,
            regime_color="green" if lane == "🟢 ATTACK" else "yellow" if lane == "🟡 ENGAGE" else "gray",
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=drivers[:3],
            narrative=narrative,
            lifecycle="CONFIRMED" if entry > 0 else "EMERGING" if bias == "WATCH" else "WAITING",
            flow_score=0.5
        )

    def _build_narrative(self, lane, score, gate_reason, is_pullback, asset_type):
        if gate_reason: return f"⚠️ {gate_reason}. {asset_type} Safety Triggered."
        if is_pullback and lane == "🟡 ENGAGE": return f"Dip Entry on {asset_type} structure."
        if lane == "🟢 ATTACK": return f"Full {asset_type} Alignment Confirmed."
        if lane == "🟡 ENGAGE": return f"Trend Confirmed ({score}%). Friction Monitor Active."
        if lane == "🟠 PREPARE": return "Energy building. Awaiting momentum."
        return "Market idle. Scanning for vectors."

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