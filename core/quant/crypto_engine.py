from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v31.1 – CLEAN UI EDITION

    LOGIC: v31 Titanium (Anti-Lag + Elasticity)
    OUTPUT: Ultra-Clean Retail Terms (No Brackets, No Physics Jargon)
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 50
        self.BASE_RISK = 0.015
        self.MAX_STRETCH = 0.03

    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None):
        price = 0.0

        try:
            if len(df) < 60:
                return self._neutral(0, "Insufficient History")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]
            price = float(close.iloc[-1])

            # ===================================================================
            # 1. CORE MOMENTUM (Anti-Lag Force)
            # ===================================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            velocity = close.diff() / (sigma + 1e-9)
            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            # Fast Smoothing (2-period) to catch moves early
            force = (mass * velocity).rolling(2).mean()
            force_now = float(force.iloc[-1])

            # Acceleration (Are we speeding up or slowing down?)
            acceleration = force.diff(2).iloc[-1]

            # ===================================================================
            # 2. ENERGY GAUGE (Hidden RSI)
            # ===================================================================
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            energy_reserve = 100 - (100 / (1 + rs))
            current_energy = float(energy_reserve.iloc[-1])

            # ===================================================================
            # 3. STRUCTURE & ELASTICITY
            # ===================================================================
            eq = close.ewm(span=self.STRUCT_LEN).mean().iloc[-1]
            stretch_pct = (price - eq) / eq

            is_overstretched_long = stretch_pct > self.MAX_STRETCH
            is_overstretched_short = stretch_pct < -self.MAX_STRETCH

            bull_struct = price > eq
            bear_struct = price < eq

            # ===================================================================
            # 4. TRAP DETECTION
            # ===================================================================
            body = abs(close.iloc[-1] - open_p.iloc[-1])
            wick_upper = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            wick_lower = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]

            is_wick_trap = False
            if force_now > 0 and wick_upper > (body * 1.2): is_wick_trap = True
            if force_now < 0 and wick_lower > (body * 1.2): is_wick_trap = True

            is_exhaustion = False
            if abs(force_now) > 1.5 and acceleration < 0:
                is_exhaustion = True

            # ===================================================================
            # 5. REGIME DETECTION
            # ===================================================================
            if abs(force_now) < 0.6:
                regime = "COMPRESSION"
            elif abs(force_now) > 2.0 and acceleration > 0:
                regime = "EXPANSION"
            elif abs(force_now) > 0.8:
                regime = "TREND"
            else:
                regime = "IDLE"

            # ===================================================================
            # 6. SIGNAL GENERATION
            # ===================================================================
            bias = "HOLD"
            lane = "⚫ HOLD"
            score = 50

            vol_scalar = float(mass.iloc[-1])

            valid_long_energy = current_energy < 75
            valid_short_energy = current_energy > 25
            valid_vol = vol_scalar > 1.0

            # LONG
            if (regime in ["TREND", "EXPANSION"] and
                    force_now > 0.8 and
                    acceleration > -0.1 and
                    bull_struct and
                    not is_overstretched_long and
                    valid_long_energy and
                    valid_vol and
                    not is_wick_trap and
                    not is_exhaustion):

                bias = "LONG"
                score = 85
                lane = "🟢 SNIPER" if regime == "TREND" else "🚀 BREAKOUT"

            # SHORT
            elif (regime in ["TREND", "EXPANSION"] and
                  force_now < -0.8 and
                  acceleration < 0.1 and
                  bear_struct and
                  not is_overstretched_short and
                  valid_short_energy and
                  valid_vol and
                  not is_wick_trap and
                  not is_exhaustion):

                bias = "SHORT"
                score = 85
                lane = "🟢 SNIPER" if regime == "TREND" else "🚀 BREAKOUT"

            # WATCH
            elif regime == "COMPRESSION":
                bias = "WATCH"
                score = 60
                lane = "🟠 BUILDING"

        except Exception as e:
            log.error(f"Engine Crash: {e}")
            return self._neutral(price, "System Error")

        # =======================================================================
        # 7. OUTPUTS
        # =======================================================================
        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1
            stop_dist = current_sigma * 1.5
            stop = price - (direction * stop_dist)
            t1 = price + (direction * stop_dist * 2.0)
            t2 = price + (direction * stop_dist * 4.0)
            t3 = price + (direction * stop_dist * 8.0)
            risk_pct = self.BASE_RISK * min(1.5, abs(force_now))
            rr = 2.0

        # =======================================================================
        # 8. CLEAN LOGIC VECTORS (NO BRACKETS, NO PHYSICS)
        # =======================================================================
        top_features = []

        whale_z = float(vol_scalar - 1.0)
        whale_active = abs(whale_z) > 1.5

        if bias in ["LONG", "SHORT"]:
            # 1. Momentum Vector
            desc = "Accelerating Bullish Momentum" if bias == "LONG" else "Accelerating Bearish Momentum"
            top_features.append({"desc": desc, "importance": 95})

            # 2. Structure Vector
            desc = "Clean Breakout Structure" if regime == "EXPANSION" else "Trend Following Structure"
            top_features.append({"desc": desc, "importance": 90})

            # 3. Volume Vector
            if whale_active:
                top_features.append({"desc": "Heavy Institutional Volume", "importance": 85})
            else:
                top_features.append({"desc": "Healthy Volume Inflow", "importance": 80})

        elif bias == "WATCH":
            top_features.append({"desc": "Volatility Compression", "importance": 80})
            top_features.append({"desc": "Awaiting Kinetic Impulse", "importance": 70})

        else:  # HOLD
            # Clean reasons for holding (No brackets)
            if is_overstretched_long or is_overstretched_short:
                top_features.append({"desc": "Price Overextended", "importance": 90})
            elif not valid_long_energy or not valid_short_energy:
                top_features.append({"desc": "Momentum Exhausted", "importance": 90})
            elif is_wick_trap:
                top_features.append({"desc": "Wick Rejection Detected", "importance": 80})
            elif is_exhaustion:
                top_features.append({"desc": "Momentum Deceleration", "importance": 80})
            else:
                top_features.append({"desc": "Market Noise", "importance": 50})

        # Failsafe
        if not top_features: top_features.append({"desc": "Analyzing Data", "importance": 0})
        top_features = top_features[:3]

        return SimpleNamespace(
            bias=bias, lane=lane, score=score, price=price,
            entry=price if bias in ["LONG", "SHORT"] else 0.0,
            stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=round(rr, 2), risk_pct=round(risk_pct * 100, 2),
            regime=regime, regime_color="green" if bias != "HOLD" else "gray",
            whale_state="ACTIVE" if whale_active else "BASELINE",
            top_features=top_features,
            narrative="Setup Validated" if bias != "HOLD" else "Waiting for optimal alignment.",
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price,
            entry=0, stop=0, target1=0, target2=0, target3=0, rr_ratio=0, risk_pct=0,
            regime="NEUTRAL", regime_color="gray", whale_state="BASELINE",
            top_features=[{"desc": "Initializing...", "importance": 10}],
            narrative=reason, lifecycle="WAITING"
        )