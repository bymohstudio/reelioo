from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO KINETIC ENGINE v28.1 – RETAIL CLARITY UPDATE

    FIXES:
    ------
    ✓ DYNAMIC VECTORS: Logic tags now include real data (e.g., "3.5x Vol").
    ✓ SHORT LOGIC FIXED: Added specific vectors for Short signals (previously missing).
    ✓ RETAIL LANGUAGE: Converted "Vector Alignment" -> "Momentum Aligned".
    """

    def __init__(self):
        self.SIGMA_WINDOW = 14
        self.MASS_WINDOW = 20
        self.STRUCT_WINDOW = 20
        self.BASE_RISK = 0.01

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY", market_context: dict = None) -> SimpleNamespace:
        """
        Analyze price data with Smoothed Kinetic Vectors + Market Context.
        """
        price = 0.0

        try:
            if len(df) < 55:
                return self._neutral(0.0, "Insufficient Data (Need 55+ Epochs)")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            open_p = df["open"]
            volume = df["volume"]

            price = float(close.iloc[-1])

            # ==================================================
            # 1. KINETIC VECTOR CALCULATIONS
            # ==================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            sigma_series = tr.rolling(self.SIGMA_WINDOW).mean()
            current_sigma = sigma_series.iloc[-1]

            velocity = close.diff() / (sigma_series + 1e-9)

            mass_mean = volume.rolling(self.MASS_WINDOW).mean()
            mass = volume / (mass_mean + 1e-9)

            raw_force = mass * velocity

            # Smooth the Force
            smooth_force = raw_force.rolling(3).mean()
            force_now = smooth_force.iloc[-1]
            force_raw = raw_force.iloc[-1]
            force_decay = smooth_force.diff(3)

            # ==================================================
            # 2. EQUILIBRIUM BASELINE
            # ==================================================
            equilibrium = close.ewm(span=50, adjust=False).mean().iloc[-1]
            is_positive_alignment = price > equilibrium
            is_negative_alignment = price < equilibrium

            # ==================================================
            # 3. TOPOGRAPHY & REJECTIONS
            # ==================================================
            roll_high = high.rolling(self.STRUCT_WINDOW).max().shift(1)
            roll_low = low.rolling(self.STRUCT_WINDOW).min().shift(1)

            range_high = float(roll_high.iloc[-1])
            range_low = float(roll_low.iloc[-1])

            candle_body = abs(close.iloc[-1] - open_p.iloc[-1])
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            lower_wick = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]

            is_rejection = False
            if force_raw > 0:
                is_rejection = upper_wick > (1.5 * candle_body)
            elif force_raw < 0:
                is_rejection = lower_wick > (1.5 * candle_body)

            # ==================================================
            # 4. ANOMALY FILTERS
            # ==================================================
            candle_range = high - low
            is_wide_range = candle_range.iloc[-1] > (2.5 * current_sigma)
            mass_intensity = volume.iloc[-1] / (mass_mean.iloc[-1] + 1e-9)

            is_hollow = (is_wide_range and mass_intensity < 0.9)
            is_overextended = abs(force_now) > 3.5
            is_trap = is_hollow or is_rejection or is_overextended

            # ==================================================
            # 5. RESONANCE
            # ==================================================
            ht_velocity = close.diff(5) / (sigma_series + 1e-9)
            ht_force = (mass * ht_velocity).rolling(5).mean()
            resonance = np.sign(force_now) == np.sign(ht_force.iloc[-1])

        except Exception as e:
            log.error(f"Quant Engine Error: {e}")
            return self._neutral(price, f"Compute Error: {e}")

        # ==========================================================
        # 6. REGIME DETECTION
        # ==========================================================
        decay_val = force_decay.iloc[-1]

        if abs(force_now) < 0.6:
            regime = "COMPRESSION"
        elif abs(force_now) > 2.0 and decay_val > 0:
            regime = "EXPANSION"
        elif abs(force_now) > 0.8:
            regime = "TREND"
        elif decay_val < -0.5:
            regime = "EXHAUSTION"
        else:
            regime = "IDLE"

        # ==========================================================
        # 7. DECISION ENGINE
        # ==========================================================
        bias = "HOLD"
        lane = "⚫ HOLD"
        score = 50

        is_high_quality = mass.iloc[-1] > 1.2

        # A. TREND
        if regime == "TREND":
            if force_now > 0 and (is_positive_alignment) and resonance and not is_trap and is_high_quality:
                bias = "LONG"
                lane = "🟢 SNIPER"
                score = 85
            elif force_now < 0 and (is_negative_alignment) and resonance and not is_trap and is_high_quality:
                bias = "SHORT"
                lane = "🟢 SNIPER"
                score = 85

        # B. BREAKOUT
        elif regime == "EXPANSION":
            if force_now > 2.0 and not is_trap and is_positive_alignment:
                bias = "LONG"
                lane = "🚀 BREAKOUT"
                score = 90
            elif force_now < -2.0 and not is_trap and is_negative_alignment:
                bias = "SHORT"
                lane = "🚀 BREAKOUT"
                score = 90

        # C. RANGE
        elif regime == "COMPRESSION":
            bias = "WATCH"
            lane = "🟠 BUILDING"
            score = 60
            range_width = (range_high - range_low) / range_low

            if range_width > 0.015:
                dist_to_support = (price - range_low) / range_low
                dist_to_resist = (range_high - price) / range_high

                if dist_to_support < 0.01 and velocity.iloc[-1] > 0 and not is_negative_alignment:
                    bias = "LONG"
                    lane = "🔵 RANGE"
                    score = 75
                elif dist_to_resist < 0.01 and velocity.iloc[-1] < 0 and not is_positive_alignment:
                    bias = "SHORT"
                    lane = "🔵 RANGE"
                    score = 75

        elif regime == "EXHAUSTION":
            bias = "HOLD"
            lane = "🔴 EXHAUSTION"
            score = 45

        # ==========================================================
        # 8. EXTERNAL CORRELATION FILTER
        # ==========================================================
        is_correlation_drag = False
        if market_context and bias == "LONG":
            btc_regime = market_context.get('regime', 'NEUTRAL')
            btc_bias = market_context.get('bias', 'HOLD')
            if btc_regime == "EXHAUSTION" or btc_bias == "SHORT":
                bias = "HOLD"
                lane = "⚫ MACRO DRAG"
                score = 50
                is_correlation_drag = True

        # ==========================================================
        # 9. TARGETING
        # ==========================================================
        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1
            stop_mult = 1.0 if lane == "🔵 RANGE" else 1.5
            if regime == "EXPANSION": stop_mult = 1.8
            stop = price - (direction * current_sigma * stop_mult)

            if lane == "🔵 RANGE":
                t1 = range_high if bias == "LONG" else range_low
                t2 = t1;
                t3 = t1
            else:
                t1 = price + (direction * current_sigma * 2.0)
                t2 = price + (direction * current_sigma * 4.0)
                t3 = price + (direction * current_sigma * 7.0)

            risk_dist = abs(price - stop)
            reward_dist = abs(t1 - price)
            if risk_dist > 0: rr = reward_dist / risk_dist
            conviction = min(1.5, abs(force_now))
            risk_pct = self.BASE_RISK * conviction

        # ==========================================================
        # 10. DYNAMIC LOGIC VECTORS (RETAIL FRIENDLY)
        # ==========================================================
        whale_z = float(mass_intensity - 1.0)
        whale_state = "ACTIVE" if abs(whale_z) >= 2.0 else "BASELINE"
        whale_label = "Institutional" if whale_state == "ACTIVE" else "Standard"

        top_features = []

        # A. Warning Vectors (Overrides everything)
        if is_correlation_drag:
            top_features.append({"desc": "WARNING: Bitcoin is Dumping", "importance": 100})

        # B. Active Trade Vectors
        elif bias != "HOLD":

            # 1. Volume Logic (Dynamic)
            vol_x = mass_intensity.iloc[-1]
            if whale_state == "ACTIVE":
                top_features.append({"desc": f"Whale Volume Detected ({vol_x:.1f}x Avg)", "importance": 95})
            elif is_high_quality:
                top_features.append({"desc": f"Healthy Volume ({vol_x:.1f}x Avg)", "importance": 80})

            # 2. Structural Alignment (Dynamic)
            if bias == "LONG" and is_positive_alignment:
                top_features.append({"desc": "Uptrend Confirmed (Above 50 EMA)", "importance": 85})
            elif bias == "SHORT" and is_negative_alignment:
                top_features.append({"desc": "Downtrend Confirmed (Below 50 EMA)", "importance": 85})

            # 3. Regime Specific Logic
            if regime == "EXPANSION":
                top_features.append({"desc": f"Explosive Velocity ({force_now:.1f}σ)", "importance": 90})

            elif lane == "🔵 RANGE":
                if bias == "LONG":
                    top_features.append({"desc": "Bouncing off Range Support", "importance": 85})
                else:
                    top_features.append({"desc": "Rejecting Range Resistance", "importance": 85})

            # 4. Resonance
            if resonance:
                top_features.append({"desc": "Momentum Aligned (15m + 1H)", "importance": 80})

        # C. Hold/Watch Vectors (Why are we waiting?)
        elif bias == "HOLD":
            if is_trap:
                top_features.append({"desc": "Trap Detected: Fakeout/Wick", "importance": 70})
            elif regime == "COMPRESSION":
                top_features.append({"desc": "Market Squeezing (Wait for Break)", "importance": 50})
            elif regime == "IDLE":
                top_features.append({"desc": "Low Volatility / No Edge", "importance": 30})
            elif not is_high_quality:
                top_features.append({"desc": "Volume Too Low", "importance": 40})

        # Sort by importance
        top_features.sort(key=lambda x: x['importance'], reverse=True)
        top_features = top_features[:3]

        narrative = self._narrative(regime, bias, resonance, whale_state, is_positive_alignment)

        return SimpleNamespace(
            bias=bias, lane=lane, score=score, price=price,
            entry=price if bias in ["LONG", "SHORT"] else 0.0,
            stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=round(rr, 2), risk_pct=round(risk_pct * 100, 2),
            regime=regime, regime_color="green" if bias != "HOLD" else "gray",
            whale_state=whale_state, whale_label=whale_label, narrative=narrative,
            top_features=top_features,
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    def _narrative(self, regime, bias, resonance, whale, positive_structure):
        if bias == "HOLD":
            if regime == "COMPRESSION": return "Market is squeezing. Waiting for energy."
            if not positive_structure and regime == "TREND": return "Price below Baseline. Longs invalid."
            return "No clear edge detected."

        if regime == "EXPANSION": return "Velocity breakout initialized."
        if regime == "TREND": return "Trend alignment confirmed."
        if whale == "ACTIVE": return "Institutional mass detected."
        return "Setup valid."

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price,
            entry=0.0, stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0.0, risk_pct=0.0,
            regime="NEUTRAL", regime_color="gray",
            whale_state="BASELINE", whale_label="Normal",
            narrative=reason, lifecycle="WAITING", top_features=[]
        )