from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO PHYSICS ENGINE v20 – AUTO REGIME SWITCHER

    REGIMES
    -------
    • COMPRESSION  → Energy building
    • TREND        → Directional continuation
    • EXPANSION    → Breakout / impulse
    • EXHAUSTION   → Late / distribution

    CORE UPGRADES
    -------------
    ✓ Signed kinetic energy
    ✓ Energy decay detection
    ✓ Structural confirmation (HH / LL)
    ✓ Event rejection (liquidity traps)
    ✓ Multi-TF physics resonance
    ✓ Auto-regime-based sizing & targets
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20
        self.BASE_RISK = 0.01  # 1%

    # ==========================================================
    # MAIN
    # ==========================================================

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0

        try:
            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
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

            velocity = close.diff() / (atr + 1e-9)

            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)

            signed_ke = mass * velocity          # 🔥 signed energy
            ke_decay = signed_ke.diff(3)

            # ==================================================
            # 2. STRUCTURE (NO EMA / RSI)
            # ==================================================

            hh = high > high.rolling(self.STRUCT_LEN).max().shift(1)
            hl = low > low.rolling(self.STRUCT_LEN).min().shift(1)

            lh = high < high.rolling(self.STRUCT_LEN).max().shift(1)
            ll = low < low.rolling(self.STRUCT_LEN).min().shift(1)

            structure_up = hh.iloc[-1] and hl.iloc[-1]
            structure_down = lh.iloc[-1] and ll.iloc[-1]

            # ==================================================
            # 3. EVENT REJECTION (LIQUIDITY TRAPS)
            # ==================================================

            candle_range = (high - low) / price
            vol_intensity = volume / (vol_mean + 1e-9)

            is_fake = (
                candle_range.iloc[-1] > 0.02 and
                vol_intensity.iloc[-1] < 0.8
            )

            # ==================================================
            # 4. MULTI-TF ENERGY RESONANCE
            # ==================================================

            ht_velocity = close.diff(5) / (atr + 1e-9)
            ht_ke = (mass * ht_velocity).rolling(5).mean()

            resonance = np.sign(signed_ke.iloc[-1]) == np.sign(ht_ke.iloc[-1])

        except Exception as e:
            log.error(e)
            return self._neutral(price, f"Data Error: {e}")

        # ==========================================================
        # 5. AUTO REGIME DETECTION
        # ==========================================================

        ke_now = signed_ke.iloc[-1]
        ke_prev = signed_ke.iloc[-4]
        decay = ke_decay.iloc[-1]

        regime = "IDLE"

        if abs(ke_now) < 0.5 and abs(decay) < 0.1:
            regime = "COMPRESSION"
        elif abs(ke_now) > 1.5 and decay > 0:
            regime = "EXPANSION"
        elif abs(ke_now) > 1.0 and decay >= 0:
            regime = "TREND"
        elif decay < 0:
            regime = "EXHAUSTION"

        # ==========================================================
        # 6. DECISION ENGINE (REGIME-AWARE)
        # ==========================================================

        bias = "HOLD"
        lane = "⚫ HOLD"
        score = 50

        if regime == "TREND":
            if ke_now > 0 and structure_up and not is_fake:
                bias = "LONG"
                lane = "🟢 TREND"
                score = 80 if resonance else 70
            elif ke_now < 0 and structure_down and not is_fake:
                bias = "SHORT"
                lane = "🟢 TREND"
                score = 80 if resonance else 70

        elif regime == "EXPANSION":
            if ke_now > 0 and not is_fake:
                bias = "LONG"
                lane = "🚀 BREAKOUT"
                score = 90
            elif ke_now < 0 and not is_fake:
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
        # 7. ADAPTIVE POSITION SIZING
        # ==========================================================

        atr_now = atr.iloc[-1]
        stop = t1 = t2 = t3 = 0.0
        rr = 0.0
        risk_pct = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1

            stop_mult = {
                "TREND": 1.5,
                "EXPANSION": 2.0,
                "COMPRESSION": 1.8
            }.get(regime, 1.6)

            stop = price - direction * atr_now * stop_mult

            target_mult = {
                "TREND": (2.0, 3.5, 5.0),
                "EXPANSION": (3.0, 5.0, 8.0)
            }.get(regime, (2.0, 3.0, 4.0))

            t1 = price + direction * atr_now * target_mult[0]
            t2 = price + direction * atr_now * target_mult[1]
            t3 = price + direction * atr_now * target_mult[2]

            conviction = min(1.5, abs(ke_now))
            risk_pct = self.BASE_RISK * conviction * (1.2 if resonance else 1.0)

            rr = abs(t1 - price) / abs(price - stop)

        # ==========================================================
        # 8. EXPLAINABILITY (UI SAFE)
        # ==========================================================

        top_features = [
            {"desc": f"Regime: {regime}", "importance": 100},
            {"desc": "Directional Energy", "importance": int(min(100, abs(ke_now) * 30))}
        ]

        if resonance:
            top_features.append({"desc": "Multi-TF Resonance", "importance": 90})
        if is_fake:
            top_features.append({"desc": "Liquidity Trap Rejected", "importance": 95})

        top_features = sorted(top_features, key=lambda x: x["importance"], reverse=True)[:3]

        narrative = self._narrative(regime, bias, resonance)

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
            expected_duration="2h",
            regime=regime,
            regime_color="green" if bias != "HOLD" else "gray",
            top_features=top_features,
            narrative=narrative,
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _narrative(self, regime, bias, resonance):
        if regime == "COMPRESSION":
            return "Energy compressing"
        if regime == "EXHAUSTION":
            return "Momentum fading"
        if regime == "EXPANSION":
            return "Breakout energy detected"
        if resonance:
            return "Multi-TF energy alignment"
        return "Directional energy confirmed"

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD",
            lane="⚫ HOLD",
            score=50,
            price=price,
            entry=0.0,
            stop=0.0,
            target1=0.0,
            target2=0.0,
            target3=0.0,
            rr_ratio=0.0,
            risk_pct=0.0,
            expected_duration="--",
            regime="NEUTRAL",
            regime_color="gray",
            top_features=[{"desc": "No Trade", "importance": 50}],
            narrative=reason,
            lifecycle="WAITING"
        )
