from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO PHYSICS ENGINE v26 – PROFESSIONAL GRADE

    UPDATES (v26)
    -------------
    ✓ MEAN REVERSION: Now trades the "Chop" (Buys Support / Sells Resistance) in Compression.
    ✓ MARKET CONTEXT: Accepts BTC data to filter bad Altcoin Longs.
    ✓ RETAIL NARRATIVE: Clear logic vectors for Range trades and BTC warnings.
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20  # Lookback for Range Support/Resistance
        self.BASE_RISK = 0.01

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY", market_context: dict = None) -> SimpleNamespace:
        """
        Analyze price data with Physics + Market Context.
        :param market_context: Dict containing BTC state (e.g. {'bias': 'SHORT', 'regime': 'EXHAUSTION'})
        """
        price = 0.0

        try:
            if len(df) < 50:
                return self._neutral(0.0, "Insufficient Data")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            open_p = df["open"]
            volume = df["volume"]

            price = float(close.iloc[-1])

            # ==================================================
            # 1. TRUE PHYSICS
            # ==================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            atr = tr.rolling(self.ATR_LEN).mean()
            current_atr = atr.iloc[-1]

            velocity = close.diff() / (atr + 1e-9)

            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            signed_ke = mass * velocity
            ke_decay = signed_ke.diff(3)

            # ==================================================
            # 2. STRUCTURE & WICKS
            # ==================================================
            # Calculate Range Highs/Lows for Mean Reversion
            roll_high = high.rolling(self.STRUCT_LEN).max().shift(1)
            roll_low = low.rolling(self.STRUCT_LEN).min().shift(1)

            range_high_val = float(roll_high.iloc[-1])
            range_low_val = float(roll_low.iloc[-1])

            hh = high > roll_high
            hl = low > roll_low
            lh = high < roll_high
            ll = low < roll_low

            structure_up = hh.iloc[-1] and hl.iloc[-1]
            structure_down = lh.iloc[-1] and ll.iloc[-1]

            # Wick Rejection (SFP)
            candle_body = abs(close.iloc[-1] - open_p.iloc[-1])
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            lower_wick = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]

            is_wick_rejection = False
            if signed_ke.iloc[-1] > 0:
                is_wick_rejection = upper_wick > (1.8 * candle_body)
            elif signed_ke.iloc[-1] < 0:
                is_wick_rejection = lower_wick > (1.8 * candle_body)

            # ==================================================
            # 3. FILTERS (SNIPER LOGIC)
            # ==================================================
            candle_range = high - low
            is_wide_range = candle_range.iloc[-1] > (2.5 * current_atr)
            vol_intensity = volume / (vol_mean + 1e-9)

            # Filter 1: Fakeouts
            is_fake = (is_wide_range and vol_intensity.iloc[-1] < 0.8)

            # Filter 2: Over-Extension
            is_overextended = abs(signed_ke.iloc[-1]) > 3.0

            # Combined Trap Logic
            is_trap = is_fake or is_wick_rejection or is_overextended

            # ==================================================
            # 4. RESONANCE
            # ==================================================
            ht_velocity = close.diff(5) / (atr + 1e-9)
            ht_ke = (mass * ht_velocity).rolling(5).mean()
            resonance = np.sign(signed_ke.iloc[-1]) == np.sign(ht_ke.iloc[-1])

        except Exception as e:
            log.error(f"Quant Engine Error: {e}")
            return self._neutral(price, f"Engine Error: {e}")

        # ==========================================================
        # 5. REGIME DETECTION
        # ==========================================================
        ke_now = signed_ke.iloc[-1]
        decay = ke_decay.iloc[-1]

        if abs(ke_now) < 0.5 and abs(decay) < 0.1:
            regime = "COMPRESSION"
        elif abs(ke_now) > 1.5 and decay > 0:
            regime = "EXPANSION"
        elif abs(ke_now) > 0.8 and decay >= -0.2:
            regime = "TREND"
        elif decay < -0.5:
            regime = "EXHAUSTION"
        else:
            regime = "IDLE"

        # ==========================================================
        # 6. DECISION ENGINE
        # ==========================================================
        bias = "HOLD"
        lane = "⚫ HOLD"
        score = 50

        # Strict Volume Filter
        is_high_quality = mass.iloc[-1] > 1.0

        # --- A. TREND LOGIC (Breakouts) ---
        if regime == "TREND":
            if ke_now > 0 and (structure_up or resonance) and not is_trap and is_high_quality:
                bias = "LONG"
                lane = "🟢 SNIPER"
                score = 85
            elif ke_now < 0 and (structure_down or resonance) and not is_trap and is_high_quality:
                bias = "SHORT"
                lane = "🟢 SNIPER"
                score = 85

        elif regime == "EXPANSION":
            if ke_now > 1.2 and not is_trap:
                bias = "LONG"
                lane = "🚀 BREAKOUT"
                score = 90
            elif ke_now < -1.2 and not is_trap:
                bias = "SHORT"
                lane = "🚀 BREAKOUT"
                score = 90

        # --- B. MEAN REVERSION LOGIC (Trading the Chop) ---
        elif regime == "COMPRESSION":
            bias = "WATCH"
            lane = "🟠 BUILDING"
            score = 60

            # Check if Range is wide enough to trade (>1.5%)
            range_width = (range_high_val - range_low_val) / range_low_val

            if range_width > 0.015:
                # Calc distance to support/resistance
                dist_to_support = (price - range_low_val) / range_low_val
                dist_to_resist = (range_high_val - price) / range_high_val

                # Buy Support: Price near low + Velocity turning UP
                if dist_to_support < 0.01 and velocity.iloc[-1] > 0:
                    bias = "LONG"
                    lane = "🔵 RANGE"
                    score = 75

                # Sell Resistance: Price near high + Velocity turning DOWN
                elif dist_to_resist < 0.01 and velocity.iloc[-1] < 0:
                    bias = "SHORT"
                    lane = "🔵 RANGE"
                    score = 75

        elif regime == "EXHAUSTION":
            bias = "HOLD"
            lane = "🔴 EXHAUSTION"
            score = 45

        # ==========================================================
        # 7. BTC CORRELATION FILTER (The King Factor)
        # ==========================================================
        is_btc_drag = False

        if market_context and bias == "LONG":
            btc_regime = market_context.get('regime', 'NEUTRAL')
            btc_bias = market_context.get('bias', 'HOLD')

            # Rule: If BTC is Crashing or Exhausted, kill Altcoin Longs
            if btc_regime == "EXHAUSTION" or btc_bias == "SHORT":
                bias = "HOLD"
                lane = "⚫ BTC DRAG"
                score = 50
                is_btc_drag = True

        # ==========================================================
        # 8. ADAPTIVE SIZING
        # ==========================================================
        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1

            # Tighter stops for Range trades, Looser for Trend
            stop_mult = 1.0 if lane == "🔵 RANGE" else 1.5
            if regime == "EXPANSION": stop_mult = 1.8

            stop = price - (direction * current_atr * stop_mult)

            # Targets based on Lane
            if lane == "🔵 RANGE":
                # Target is the other side of the range
                t1 = range_high_val if bias == "LONG" else range_low_val
                t2 = t1
                t3 = t1
            else:
                # Trend Targets
                target_mult = (1.5, 3.0, 6.0)
                t1 = price + (direction * current_atr * target_mult[0])
                t2 = price + (direction * current_atr * target_mult[1])
                t3 = price + (direction * current_atr * target_mult[2])

            risk_dist = abs(price - stop)
            reward_dist = abs(t1 - price)
            if risk_dist > 0:
                rr = reward_dist / risk_dist

            conviction = min(1.5, abs(ke_now))
            risk_pct = self.BASE_RISK * conviction

        # ==========================================================
        # 9. OUTPUT & RETAIL NARRATIVE VECTORS
        # ==========================================================
        whale_z = float(vol_intensity.iloc[-1] - 1.0)
        if abs(whale_z) >= 2.0:
            whale_label = "Institutional"
            whale_state = "ACTIVE"
        elif abs(whale_z) >= 0.8:
            whale_label = "Elevated"
            whale_state = "BUILDING"
        else:
            whale_label = "Retail"
            whale_state = "BASELINE"

        top_features = []

        # --- BTC VETO ---
        if is_btc_drag:
            top_features.append({"desc": "WARNING: Bitcoin is Weak (Risk Off)", "importance": 100})

        # --- SCENARIO A: HOLD ---
        elif bias == "HOLD" or bias == "WATCH":
            if is_trap:
                if is_fake:
                    top_features.append({"desc": "Fakeout Risk (Volume Mismatch)", "importance": 65})
                elif is_wick_rejection:
                    top_features.append({"desc": "Price Rejection Detected", "importance": 65})
                elif is_overextended:
                    top_features.append({"desc": "Market Overheated (Too Late)", "importance": 65})

            elif regime == "COMPRESSION":
                top_features.append({"desc": "Squeeze Building (Wait)", "importance": 50})
            elif regime == "EXHAUSTION":
                top_features.append({"desc": "Trend Fading", "importance": 45})
            elif regime == "IDLE":
                top_features.append({"desc": "Low Volatility", "importance": 30})

            if not is_high_quality:
                top_features.append({"desc": "Weak Participation", "importance": 25})

        # --- SCENARIO B: ACTIVE TRADE ---
        else:
            # Range Logic
            if lane == "🔵 RANGE":
                top_features.append(
                    {"desc": "Valid Support Bounce" if bias == "LONG" else "Resistance Rejection", "importance": 90})
                top_features.append({"desc": "Trading the Range", "importance": 80})

            # Trend Logic
            else:
                if whale_state == "ACTIVE":
                    top_features.append(
                        {"desc": "Smart Money Buying" if bias == "LONG" else "Smart Money Selling", "importance": 95})
                elif is_high_quality:
                    top_features.append({"desc": "Healthy Volume Support", "importance": 80})

                if resonance:
                    top_features.append({"desc": "Perfect Trend Alignment", "importance": 90})

                if regime == "EXPANSION":
                    top_features.append({"desc": "High Velocity Breakout", "importance": 85})

        top_features.sort(key=lambda x: x['importance'], reverse=True)
        top_features = top_features[:3]

        narrative = self._narrative(regime, bias, resonance, whale_state)

        return SimpleNamespace(
            bias=bias, lane=lane, score=score, price=price,
            entry=price if bias in ["LONG", "SHORT"] else 0.0,
            stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=round(rr, 2), risk_pct=round(risk_pct * 100, 2),
            regime=regime, regime_color="green" if bias != "HOLD" else "gray",
            whale_zscore=round(whale_z, 2), whale_label=whale_label, whale_state=whale_state,
            top_features=top_features, narrative=narrative,
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    def _narrative(self, regime, bias, resonance, whale):
        if bias == "HOLD" and regime == "COMPRESSION": return "Market is ranging. waiting for clear bounce or break."
        if regime == "COMPRESSION": return "Volatility squeezing. Energy building."
        if regime == "EXHAUSTION": return "Trend is tired. Avoid new entries."
        if regime == "EXPANSION": return "High speed move. Stops are tight."
        if whale == "ACTIVE": return "Institutions are active here."
        return "Waiting for a clear setup."

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price,
            entry=0.0, stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0.0, risk_pct=0.0,
            regime="NEUTRAL", regime_color="gray",
            whale_zscore=0.0, whale_label="Normal", whale_state="BASELINE",
            top_features=[{"desc": "No Data", "importance": 20}],
            narrative=reason, lifecycle="WAITING"
        )