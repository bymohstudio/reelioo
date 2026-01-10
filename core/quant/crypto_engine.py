from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO PHYSICS ENGINE v21 – DYNAMIC ADAPTATION

    REGIMES
    -------
    • COMPRESSION  → Energy building
    • TREND        → Directional continuation
    • EXPANSION    → Breakout / impulse
    • EXHAUSTION   → Late / distribution

    CORE UPGRADES (v21)
    -------------------
    ✓ Dynamic ATR-based Fakeout Detection (No hardcoded %)
    ✓ Live Candle Volume Normalization
    ✓ Signed kinetic energy & Decay
    ✓ Multi-TF resonance
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20
        self.BASE_RISK = 0.01  # 1% per trade

    # ==========================================================
    # MAIN ANALYZER
    # ==========================================================

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0

        try:
            # SAFETY: Ensure we have enough data
            if len(df) < 50:
                return self._neutral(0.0, "Insufficient Data")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            price = float(close.iloc[-1])

            # ==================================================
            # 1. TRUE PHYSICS (VELOCITY & MASS)
            # ==================================================

            # True Range Calculation
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(self.ATR_LEN).mean()
            current_atr = atr.iloc[-1]

            # Normalized Velocity (Price change relative to Volatility)
            velocity = close.diff() / (atr + 1e-9)

            # Normalized Mass (Volume relative to Average)
            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            # Signed Kinetic Energy (Directional Force)
            signed_ke = mass * velocity
            ke_decay = signed_ke.diff(3)

            # ==================================================
            # 2. MARKET STRUCTURE (HH / LL)
            # ==================================================

            # rolling max/min of previous candles (exclude current to avoid repainting look-ahead)
            roll_high = high.rolling(self.STRUCT_LEN).max().shift(1)
            roll_low = low.rolling(self.STRUCT_LEN).min().shift(1)

            hh = high > roll_high
            hl = low > roll_low
            lh = high < roll_high
            ll = low < roll_low

            structure_up = hh.iloc[-1] and hl.iloc[-1]
            structure_down = lh.iloc[-1] and ll.iloc[-1]

            # ==================================================
            # 3. EVENT REJECTION (LIQUIDITY TRAPS) - [FIXED]
            # ==================================================

            # OLD: candle_range > 0.02 (Broke on Alts)
            # NEW: Candle Range > 2.5x ATR (Dynamic)
            candle_range = high - low
            is_wide_range = candle_range.iloc[-1] > (2.5 * current_atr)

            # Volume Intensity
            vol_intensity = volume / (vol_mean + 1e-9)

            # A trap is: Big Move + Low Volume (Fakeout)
            is_fake = (
                    is_wide_range and
                    vol_intensity.iloc[-1] < 0.8
            )

            # ==================================================
            # 4. MULTI-TF ENERGY RESONANCE
            # ==================================================

            # Higher Timeframe Proxy (5-period lookback)
            ht_velocity = close.diff(5) / (atr + 1e-9)
            ht_ke = (mass * ht_velocity).rolling(5).mean()

            resonance = np.sign(signed_ke.iloc[-1]) == np.sign(ht_ke.iloc[-1])

        except Exception as e:
            log.error(f"Quant Engine Error: {e}")
            return self._neutral(price, f"Engine Error: {e}")

        # ==========================================================
        # 5. AUTO REGIME DETECTION
        # ==========================================================

        ke_now = signed_ke.iloc[-1]
        decay = ke_decay.iloc[-1]

        # Logic Matrix
        if abs(ke_now) < 0.5 and abs(decay) < 0.1:
            regime = "COMPRESSION"
        elif abs(ke_now) > 1.5 and decay > 0:
            regime = "EXPANSION"
        elif abs(ke_now) > 0.8 and decay >= -0.2:
            regime = "TREND"  # Lowered threshold slightly to catch trends early
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
            bias = "HOLD"  # Or "TAKE_PROFIT" if holding
            lane = "🔴 EXHAUSTION"
            score = 45

        # ==========================================================
        # 7. ADAPTIVE POSITION SIZING
        # ==========================================================

        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1

            # Dynamic Stop Multiplier based on Volatility Regime
            stop_mult = {
                "TREND": 1.5,  # Tighter stop in trends
                "EXPANSION": 2.5  # Wider stop in chaos
            }.get(regime, 1.8)

            stop = price - (direction * current_atr * stop_mult)

            # R-Multiples for Targets
            target_mult = {
                "TREND": (2.0, 3.5, 6.0),
                "EXPANSION": (1.5, 3.0, 5.0)  # Take profit faster in expansions
            }.get(regime, (2.0, 3.0, 4.0))

            t1 = price + (direction * current_atr * target_mult[0])
            t2 = price + (direction * current_atr * target_mult[1])
            t3 = price + (direction * current_atr * target_mult[2])

            # Calculate Risk/Reward
            risk_dist = abs(price - stop)
            reward_dist = abs(t1 - price)
            if risk_dist > 0:
                rr = reward_dist / risk_dist

            # Size based on conviction
            conviction = min(1.5, abs(ke_now))
            risk_pct = self.BASE_RISK * conviction * (1.2 if resonance else 1.0)

        # ==========================================================
        # 8. WHALE ACTIVITY METRICS
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

        # ==========================================================
        # 9. EXPLAINABILITY VECTORS
        # ==========================================================

        top_features = [
            {"desc": f"Regime: {regime}", "importance": 100},
            {"desc": "Momentum Velocity", "importance": int(min(100, abs(ke_now) * 30))}
        ]

        if resonance:
            top_features.append({"desc": "Multi-Timeframe Alignment", "importance": 90})
        if is_fake:
            top_features.append({"desc": "Liquidity Trap Rejected", "importance": 95})
        if whale_state == "ACTIVE":
            top_features.append({"desc": "Whale Volume Detected", "importance": 85})

        top_features = sorted(top_features, key=lambda x: x["importance"], reverse=True)[:3]
        narrative = self._narrative(regime, bias, resonance, whale_state)

        return SimpleNamespace(
            bias=bias,
            lane=lane,
            score=score,
            price=price,

            entry=price if bias in ["LONG", "SHORT"] else 0.0,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),

            rr_ratio=round(rr, 2),
            risk_pct=round(risk_pct * 100, 2),

            regime=regime,
            regime_color="green" if bias != "HOLD" else "gray",

            whale_zscore=round(whale_z, 2),
            whale_label=whale_label,
            whale_state=whale_state,

            top_features=top_features,
            narrative=narrative,
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _narrative(self, regime, bias, resonance, whale):
        if regime == "COMPRESSION":
            return "Volatility squeezing. Energy building for next move."
        if regime == "EXHAUSTION":
            return "Trend showing weakness. Divergence detected."
        if regime == "EXPANSION":
            return "High-velocity breakout. Institutional volume spike."
        if whale == "ACTIVE":
            return "Whale absorption detected. Smart money is active."
        if resonance:
            return "Physics alignment across timeframes. High probability."
        return "Market structure confirms directional energy."

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD",
            lane="⚫ HOLD",
            score=50,
            price=price,
            entry=0.0, stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0.0, risk_pct=0.0,
            regime="NEUTRAL", regime_color="gray",
            whale_zscore=0.0, whale_label="Normal", whale_state="BASELINE",
            top_features=[{"desc": "No Signal", "importance": 50}],
            narrative=reason,
            lifecycle="WAITING"
        )