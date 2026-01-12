from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO PHYSICS ENGINE v22 – PROFITABILITY PATCH

    UPDATES (v22)
    -------------
    ✓ FIXED: Expansion Risk/Reward (Was 0.6, Now 1.2+)
    ✓ ADDED: Wick Rejection Filter (SFP Protection)
    ✓ TUNED: Tighter stops on breakouts
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20
        self.BASE_RISK = 0.01  # 1% per trade

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
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
            tr = QP = pd.concat([
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
            # 2. STRUCTURE & WICKS (SFP PROTECTION)
            # ==================================================
            roll_high = high.rolling(self.STRUCT_LEN).max().shift(1)
            roll_low = low.rolling(self.STRUCT_LEN).min().shift(1)

            hh = high > roll_high
            hl = low > roll_low
            lh = high < roll_high
            ll = low < roll_low

            # FIXED: Corrected logical AND operator
            structure_up = hh.iloc[-1] and hl.iloc[-1]
            structure_down = lh.iloc[-1] and ll.iloc[-1]

            # --- NEW: WICK REJECTION FILTER ---
            # Calculates if the candle has a massive wick (price rejection)
            candle_body = abs(close.iloc[-1] - open_p.iloc[-1])
            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            lower_wick = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]

            # If wick is 1.8x bigger than body, it's a rejection (SFP)
            is_wick_rejection = False
            if signed_ke.iloc[-1] > 0:  # Bullish signal check
                is_wick_rejection = upper_wick > (1.8 * candle_body)
            elif signed_ke.iloc[-1] < 0:  # Bearish signal check
                is_wick_rejection = lower_wick > (1.8 * candle_body)

            # ==================================================
            # 3. FAKEOUT FILTER
            # ==================================================
            candle_range = high - low
            is_wide_range = candle_range.iloc[-1] > (2.5 * current_atr)
            vol_intensity = volume / (vol_mean + 1e-9)

            # A trap is: (Big Move + Low Volume) OR (Wick Rejection)
            is_fake = (
                    (is_wide_range and vol_intensity.iloc[-1] < 0.8) or
                    is_wick_rejection
            )

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

        if regime == "TREND":
            if ke_now > 0 and (structure_up or resonance) and not is_fake:
                bias = "LONG"
                lane = "🟢 TREND"
                score = 80 if resonance else 75
            elif ke_now < 0 and (structure_down or resonance) and not is_fake:
                bias = "SHORT"
                lane = "🟢 TREND"
                score = 80 if resonance else 75

        elif regime == "EXPANSION":
            # Breakouts require higher velocity confirmation
            if ke_now > 1.2 and not is_fake:
                bias = "LONG"
                lane = "🚀 BREAKOUT"
                score = 90
            elif ke_now < -1.2 and not is_fake:
                bias = "SHORT"
                lane = "🚀 BREAKOUT"
                score = 90

        elif regime == "COMPRESSION":
            bias = "WATCH"
            lane = "🟠 BUILDING"
            score = 60

        elif regime == "EXHAUSTION":
            bias = "HOLD"
            lane = "🔴 EXHAUSTION"
            score = 45

        # ==========================================================
        # 7. ADAPTIVE SIZING (PROFITABILITY FIX)
        # ==========================================================
        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1

            # --- FIXED STOP LOSS MULTIPLIERS ---
            stop_mult = {
                "TREND": 1.5,
                "EXPANSION": 1.8
            }.get(regime, 1.8)

            stop = price - (direction * current_atr * stop_mult)

            # --- FIXED TARGET MULTIPLIERS (Positive R:R) ---
            target_mult = {
                "TREND": (2.0, 3.5, 6.0),
                "EXPANSION": (2.2, 4.0, 7.0)
            }.get(regime, (2.0, 3.0, 4.0))

            t1 = price + (direction * current_atr * target_mult[0])
            t2 = price + (direction * current_atr * target_mult[1])
            t3 = price + (direction * current_atr * target_mult[2])

            risk_dist = abs(price - stop)
            reward_dist = abs(t1 - price)
            if risk_dist > 0:
                rr = reward_dist / risk_dist

            conviction = min(1.5, abs(ke_now))
            risk_pct = self.BASE_RISK * conviction * (1.2 if resonance else 1.0)

        # ==========================================================
        # 8. WHALE METRICS & OUTPUT
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

        top_features = [
            {"desc": f"Regime: {regime}", "importance": 100},
            {"desc": "Momentum Velocity", "importance": int(min(100, abs(ke_now) * 30))}
        ]
        if resonance: top_features.append({"desc": "Multi-TF Resonance", "importance": 90})
        if is_wick_rejection: top_features.append({"desc": "Wick Rejection Filter", "importance": 95})

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
        if regime == "COMPRESSION": return "Volatility squeezing. Energy building."
        if regime == "EXHAUSTION": return "Trend fading. Taking defensive stance."
        if regime == "EXPANSION": return "High-velocity breakout. Stops tightened."
        if whale == "ACTIVE": return "Institutional volume detected."
        if resonance: return "Multi-TF alignment confirmed."
        return "Market structure confirms direction."

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price,
            entry=0.0, stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0.0, risk_pct=0.0,
            regime="NEUTRAL", regime_color="gray",
            whale_zscore=0.0, whale_label="Normal", whale_state="BASELINE",
            top_features=[{"desc": "No Signal", "importance": 50}],
            narrative=reason, lifecycle="WAITING"
        )