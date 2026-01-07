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
    REELIOO QUANT PHYSICS ENGINE (v11.1 – Retail Standard)

    - TERMINOLOGY UPDATE: "Attack/Engage" replaced with "Buy/Sell/Watch/Hold".
    - LOGIC: Regime-Aware Physics (Macro/Beta/Impulse profiles).
    """

    def __init__(self):
        self.SIGMOID_K = 0.5
        log.info("🚀 QuantPhysicsEngine v11.1 (Retail Standard) Online")

    def _get_asset_profile(self, symbol: str):
        s = symbol.upper()

        # 1. MACRO LEADERS (BTC, ETH) - Strict, Needs Stability
        if any(x in s for x in ['BTC', 'ETH', 'BNB']):
            return {
                'type': 'MACRO',
                'stability_window': 6,
                'min_stability': 0.7,
                'vol_gate': 0.04,
                'penalty_mult': 12.0,
                'strong_thresh': 78  # Threshold for "STRONG BUY/SELL"
            }

        # 2. IMPULSE / MEME (PEPE, WIF) - Fast, High Volatility Allowed
        elif any(x in s for x in ['PEPE', 'WIF', 'DOGE', 'SHIB', 'BONK', 'FLOKI']):
            return {
                'type': 'IMPULSE',
                'stability_window': 3,
                'min_stability': 0.5,
                'vol_gate': 0.08,
                'penalty_mult': 6.0,
                'strong_thresh': 72
            }

        # 3. BETA (SOL, AVAX) - Standard Trenders
        else:
            return {
                'type': 'BETA',
                'stability_window': 5,
                'min_stability': 0.6,
                'vol_gate': 0.06,
                'penalty_mult': 10.0,
                'strong_thresh': 75
            }

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        symbol_name = str(df.get("symbol", "UNKNOWN"))
        profile = self._get_asset_profile(symbol_name)

        try:
            # 1. Data Enrichment
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

            # Dynamic Lookback Window
            window_size = profile['stability_window']
            recent_window = df.iloc[-window_size:]
            last = df.iloc[-1]

        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # ==========================================
        # PHASE 1: PHYSICS & STABILITY
        # ==========================================

        # Trend Vector
        current_trend = cap(last.get("ema_diff", 0) * 100) * 2.0

        # Stability Check
        trend_sign = np.sign(current_trend)
        agreement_count = sum(np.sign(row) == trend_sign for row in recent_window["ema_diff"])
        stability_ratio = agreement_count / window_size

        trend_alpha = current_trend * stability_ratio

        # Confirmation Vectors
        whale_z = cap(float(last.get("whale_z", 0))) * 1.0
        kinetic = cap(float(last.get("kinetic_energy", 0))) * 1.5

        # Pullback / Mismatch Logic
        mismatch_penalty = 0
        is_pullback = False

        if np.sign(trend_alpha) != np.sign(kinetic) and abs(trend_alpha) > 0.5:
            is_pullback = True
            atr = float(last.get("atr_14", 1.0))
            current_move_pct = abs(velocity.iloc[-1]) / price
            volatility_ratio = current_move_pct / (atr / price + 0.0001)

            mismatch_penalty = profile['penalty_mult'] * max(1.0, min(3.0, volatility_ratio))

        raw_alpha = trend_alpha + whale_z + kinetic
        raw_score = self._sigmoid(raw_alpha)

        # ==========================================
        # PHASE 2: GATES
        # ==========================================
        gate_status = "OPEN"
        gate_reason = ""

        # Temporal Stability Gate
        if stability_ratio < profile['min_stability']:
            gate_status = "CLOSED"
            gate_reason = "Trend Unstable"

        # Friction Gate
        avg_friction = df['friction_coeff'].rolling(20).mean().iloc[-1]
        if pd.isna(avg_friction): avg_friction = 0

        friction_limit = 2.2 if profile['type'] == 'IMPULSE' else 1.8
        if last['friction_coeff'] > (avg_friction * friction_limit) and avg_friction > 0:
            gate_status = "CLOSED"
            gate_reason = "High Friction"

        # Volatility Gate
        if float(last.get("atr_pct", 0)) > profile['vol_gate']:
            gate_status = "CLOSED"
            gate_reason = "Max Volatility"

        # Risk Protocol
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        kill_switch = False
        if abs(shock) > 2.8:
            kill_switch = True
            gate_status = "CLOSED"
            gate_reason = "Black Swan"

        # ==========================================
        # PHASE 3: RETAIL LANE LOGIC
        # ==========================================
        lane = "⚫ HOLD"
        bias = "HOLD"

        if gate_status == "CLOSED":
            display_score = 50
        else:
            base_score = int(50 + (raw_score - 50) * 0.95)
            display_score = int(base_score - mismatch_penalty)
            display_score = max(0, min(100, display_score))

            # Dynamic Classification based on Score
            if display_score >= profile['strong_thresh']:
                lane = "🟢 STRONG BUY"
                bias = "LONG"
            elif display_score <= (100 - profile['strong_thresh']):
                lane = "🟢 STRONG SELL"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= 65:
                lane = "🟡 BUY"
                bias = "LONG"
            elif display_score <= 35:
                lane = "🟡 SELL"
                bias = "SHORT"
                display_score = 100 - display_score
            elif display_score >= 50:
                lane = "🟠 WATCH"
                bias = "WATCH"
            else:
                lane = "🟠 WATCH"
                bias = "WATCH"
                display_score = 100 - display_score

            if display_score < 50:
                lane = "⚫ HOLD"
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
                drivers.append({"desc": "Volume Surge", "importance": 90})
            if gate_status == "CLOSED": drivers.append({"desc": f"Blocked: {gate_reason}", "importance": 100})

        narrative = self._build_narrative(lane, display_score, gate_reason, is_pullback, profile['type'])

        # Retail Friendly Regime
        regime_label = "TRENDING" if abs(kinetic) > 1.2 else "RANGING"

        return SimpleNamespace(
            bias=bias, lane=lane, score=display_score, price=price,
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

    def _build_narrative(self, lane, score, gate_reason, is_pullback, asset_type):
        if gate_reason: return f"⚠️ {gate_reason}. Trade blocked for safety."
        if is_pullback and "BUY" in lane: return f"Dip Entry on {asset_type} structure."
        if "STRONG" in lane: return f"High Conviction {asset_type} Setup. Full Alignment."
        if "BUY" in lane or "SELL" in lane: return f"Trend Confirmed ({score}%). Standard Entry."
        if "WATCH" in lane: return "Setup developing. Waiting for momentum."
        return "No Edge. Capital Preserved."

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