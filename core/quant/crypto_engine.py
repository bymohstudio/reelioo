from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging
from core.services.marketdata_service import MarketService

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v32.2 – DYNAMIC VECTORS EDITION

    UPGRADE:
    --------
    Logic Vectors are now MATHEMATICALLY CALCULATED integers, not hardcoded.
    - Compression % based on inverse Force.
    - Impulse % based on Volume Scalar.
    - Wall % based on exact Order Book Imbalance.
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 50
        self.BASE_RISK = 0.015
        self.MAX_STRETCH = 0.03

    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None, symbol=None):
        price = 0.0

        try:
            if len(df) < 60: return self._neutral(0, "Insufficient History")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]
            price = float(close.iloc[-1])

            # ===================================================================
            # 1. PHYSICS VECTORS
            # ===================================================================
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            sigma = tr.rolling(self.ATR_LEN).mean()
            current_sigma = float(sigma.iloc[-1])

            velocity = close.diff() / (sigma + 1e-9)
            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            force = (mass * velocity).rolling(2).mean()
            force_now = float(force.iloc[-1])
            acceleration = force.diff(2).iloc[-1]

            # ===================================================================
            # 2. ENERGY & ELASTICITY
            # ===================================================================
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            energy_reserve = 100 - (100 / (1 + rs))
            current_energy = float(energy_reserve.iloc[-1])

            eq = close.ewm(span=self.STRUCT_LEN).mean().iloc[-1]
            stretch_pct = (price - eq) / eq

            is_overstretched_long = stretch_pct > self.MAX_STRETCH
            is_overstretched_short = stretch_pct < -self.MAX_STRETCH
            bull_struct = price > eq
            bear_struct = price < eq

            # ===================================================================
            # 3. TRAP DETECTION
            # ===================================================================
            body = abs(close.iloc[-1] - open_p.iloc[-1])
            wick_upper = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            wick_lower = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]

            is_wick_trap = False
            if force_now > 0 and wick_upper > (body * 1.2): is_wick_trap = True
            if force_now < 0 and wick_lower > (body * 1.2): is_wick_trap = True

            is_exhaustion = (abs(force_now) > 1.5 and acceleration < 0)

            # ===================================================================
            # 4. REGIME
            # ===================================================================
            if abs(force_now) < 0.8:
                regime = "COMPRESSION"
            elif abs(force_now) > 2.0 and acceleration > 0:
                regime = "EXPANSION"
            elif abs(force_now) > 1.0:
                regime = "TREND"
            else:
                regime = "IDLE"

            # ===================================================================
            # 5. SIGNAL GENERATION (STRICT IMPULSE)
            # ===================================================================
            bias = "HOLD"
            lane = "⚫ HOLD"
            score = 50

            vol_scalar = float(mass.iloc[-1])
            valid_long_energy = current_energy < 75
            valid_short_energy = current_energy > 25
            valid_vol = vol_scalar > 1.0

            MIN_FORCE = 1.2
            MIN_ACCEL = 0.05

            # LONG
            if (regime in ["TREND", "EXPANSION"] and force_now > MIN_FORCE and acceleration > MIN_ACCEL and
                    bull_struct and not is_overstretched_long and valid_long_energy and valid_vol and
                    not is_wick_trap and not is_exhaustion):
                bias = "LONG";
                score = 85
                lane = "🟢 SNIPER" if regime == "TREND" else "🚀 BREAKOUT"

            # SHORT
            elif (regime in ["TREND", "EXPANSION"] and force_now < -MIN_FORCE and acceleration < -MIN_ACCEL and
                  bear_struct and not is_overstretched_short and valid_short_energy and valid_vol and
                  not is_wick_trap and not is_exhaustion):
                bias = "SHORT";
                score = 85
                lane = "🟢 SNIPER" if regime == "TREND" else "🚀 BREAKOUT"

            elif regime == "COMPRESSION":
                bias = "WATCH";
                score = 60
                lane = "🟠 BUILDING"

            # ===================================================================
            # 6. ORDER FLOW CHECK
            # ===================================================================
            obi_score = 0.0
            blocked_by_of = False

            if bias in ["LONG", "SHORT"] and symbol:
                try:
                    data = MarketService.get_order_book_snapshot(symbol)
                    if data:
                        bids = np.array(data['bids'], dtype=float)
                        asks = np.array(data['asks'], dtype=float)
                        bid_vol = np.sum(bids[:, 1])
                        ask_vol = np.sum(asks[:, 1])
                        obi_score = (bid_vol - ask_vol) / (bid_vol + ask_vol)

                        if bias == "LONG" and obi_score < -0.25:
                            bias = "HOLD";
                            lane = "⚫ BLOCKED";
                            blocked_by_of = True;
                            score = 50
                        elif bias == "SHORT" and obi_score > 0.25:
                            bias = "HOLD";
                            lane = "⚫ BLOCKED";
                            blocked_by_of = True;
                            score = 50
                except:
                    pass

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
            stop_dist = current_sigma * 1.2
            stop = price - (direction * stop_dist)
            t1 = price + (direction * stop_dist * 2.0)
            t2 = price + (direction * stop_dist * 4.0)
            t3 = price + (direction * stop_dist * 8.0)
            risk_pct = self.BASE_RISK * min(1.5, abs(force_now))
            rr = 2.0

        # =======================================================================
        # 8. DYNAMIC LOGIC VECTORS (REAL MATH)
        # =======================================================================
        top_features = []
        whale_active = abs(vol_scalar - 1.0) > 1.5

        # --- Helper for mapping ranges to percentages ---
        def calc_pct(val, min_v, max_v, target_min=60, target_max=99):
            norm = (abs(val) - min_v) / (max_v - min_v)
            norm = max(0.0, min(1.0, norm))  # Clamp 0-1
            return int(target_min + (norm * (target_max - target_min)))

        if blocked_by_of:
            # Wall strength based on OBI score (-1.0 to 1.0)
            wall_strength = calc_pct(obi_score, 0.25, 0.8, 80, 100)
            desc = "Order Book Sell Wall" if obi_score < 0 else "Order Book Buy Wall"
            top_features.append({"desc": desc, "importance": wall_strength})

        elif bias in ["LONG", "SHORT"]:
            # Momentum Importance based on Force (1.2 to 3.0)
            mom_imp = calc_pct(force_now, 1.2, 3.0, 85, 99)
            desc = "High Velocity Impulse" if bias == "LONG" else "High Velocity Dump"
            top_features.append({"desc": desc, "importance": mom_imp})

            # Order Flow Support based on OBI (0.1 to 0.5)
            of_imp = calc_pct(obi_score, 0.1, 0.5, 75, 95)
            if bias == "LONG" and obi_score > 0.1:
                top_features.append({"desc": "Strong Bid Support", "importance": of_imp})
            elif bias == "SHORT" and obi_score < -0.1:
                top_features.append({"desc": "Strong Ask Pressure", "importance": of_imp})
            else:
                top_features.append({"desc": "Clean Market Structure", "importance": 85})  # Baseline

            # Volume Importance based on Scalar (1.0 to 4.0)
            vol_imp = calc_pct(vol_scalar, 1.0, 4.0, 70, 95)
            if whale_active:
                top_features.append({"desc": "Heavy Institutional Volume", "importance": vol_imp})
            else:
                top_features.append({"desc": "Volume Confirmation", "importance": vol_imp})

        elif bias == "WATCH":
            # Compression: Higher if force is closer to 0
            comp_imp = 100 - calc_pct(force_now, 0.0, 0.8, 0, 40)  # 100 - (0 to 40) = 100 to 60
            top_features.append({"desc": "Volatility Compression", "importance": comp_imp})

            # Kinetic Impulse Waiting: Higher if volume is building
            wait_imp = calc_pct(vol_scalar, 0.5, 2.0, 60, 90)
            top_features.append({"desc": "Awaiting Kinetic Impulse", "importance": wait_imp})

        else:  # HOLD REASONS
            if is_overstretched_long or is_overstretched_short:
                # Stretch severity (3% to 6%)
                stretch_imp = calc_pct(stretch_pct, 0.03, 0.06, 80, 100)
                top_features.append({"desc": "Price Overextended", "importance": stretch_imp})

            elif not valid_long_energy or not valid_short_energy:
                # Energy exhaustion (75 to 90 RSI)
                exh_imp = calc_pct(current_energy, 75, 90, 80, 99)
                top_features.append({"desc": "Momentum Exhausted", "importance": exh_imp})

            elif is_wick_trap:
                top_features.append(
                    {"desc": "Wick Rejection Detected", "importance": 85})  # Wicks are usually significant

            elif is_exhaustion:
                # Decel severity
                decel_imp = calc_pct(acceleration, 0.0, 0.5, 75, 95)
                top_features.append({"desc": "Momentum Deceleration", "importance": decel_imp})

            else:
                top_features.append({"desc": "Market Noise", "importance": 50})

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