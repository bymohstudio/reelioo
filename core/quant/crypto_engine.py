from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO ENGINE v30 – TITANIUM SNIPER HYBRID EDITION
    ----------------------------------------------------
    TWO MODES:
        • SWING-SNIPER → UI = Intraday (cron runs this)
        • PURE-SCALP   → UI = Scalp (manual only, ultra accurate)

    CORE UPGRADES:
        ✓ Hybrid physics (fast for alts, stable for majors)
        ✓ Signed kinetic energy
        ✓ Trap detection (wick, liquidity, exhaustion)
        ✓ Multi-TF resonance
        ✓ Volume confirmation
        ✓ Structural confirmation
        ✓ Market adaptive scoring
        ✓ Retail-friendly clean text
    """

    def __init__(self):
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20

        # Risk base for swing mode — scalp uses ultra tight mode
        self.BASE_RISK_SWING = 0.012
        self.BASE_RISK_SCALP = 0.006

    # ==========================================================================
    # MAIN ANALYSIS
    # ==========================================================================
    def analyze(self, df: pd.DataFrame, trade_style="INTRADAY", market_context=None):
        """
        trade_style:
             INTRADAY → Swing Sniper Mode
             SCALP    → Ultra Sniper Mode
        """
        price = 0.0

        try:
            if len(df) < 55:
                return self._neutral(0, "Insufficient data")

            df = df.copy()
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]
            open_p = df["open"]
            price = float(close.iloc[-1])

            # ===================================================================
            # 1. Physics: Volatility + Momentum + Volume
            # ===================================================================
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            sigma = tr.rolling(self.ATR_LEN).mean()

            velocity = close.diff() / (sigma + 1e-9)     # normalized momentum
            vol_mean = volume.rolling(self.MASS_LEN).mean()
            mass = volume / (vol_mean + 1e-9)            # volume pressure

            force = (mass * velocity).rolling(3).mean()  # smoothed signed KE
            force_now = float(force.iloc[-1])
            force_prev = float(force.iloc[-4])
            decay = force_now - force_prev

            # ===================================================================
            # 2. Structure: HH/LL clean confirmation
            # ===================================================================
            eq = close.ewm(span=50).mean().iloc[-1]
            bull_struct = price > eq
            bear_struct = price < eq

            # ===================================================================
            # 3. Liquidity Trap Detection
            # ===================================================================
            wick_upper = high.iloc[-1] - max(close.iloc[-1], open_p.iloc[-1])
            wick_lower = min(close.iloc[-1], open_p.iloc[-1]) - low.iloc[-1]
            body = abs(close.iloc[-1] - open_p.iloc[-1])

            trap = False
            if force_now > 0 and wick_upper > body * 1.5:
                trap = True
            if force_now < 0 and wick_lower > body * 1.5:
                trap = True

            wide = (high.iloc[-1] - low.iloc[-1]) > 2.2 * sigma.iloc[-1]
            hollow = (mass.iloc[-1] < 0.9)
            if wide and hollow:
                trap = True

            # ===================================================================
            # 4. Multi-TF Resonance
            # ===================================================================
            ht_vel = close.diff(5) / (sigma + 1e-9)
            ht_force = (mass * ht_vel).rolling(5).mean().iloc[-1]
            resonance = (np.sign(force_now) == np.sign(ht_force))

        except Exception as e:
            log.error(f"Engine error: {e}")
            return self._neutral(price, "System error")

        # =======================================================================
        # 5. Regime Detection
        # =======================================================================
        if abs(force_now) < 0.6:
            regime = "COMPRESSION"
        elif abs(force_now) > 2.0 and decay > 0:
            regime = "EXPANSION"
        elif abs(force_now) > 0.8:
            regime = "TREND"
        elif decay < -0.5:
            regime = "EXHAUSTION"
        else:
            regime = "IDLE"

        # =======================================================================
        # 6. MODE SWITCHING (Hybrid)
        # =======================================================================
        scalp_mode = (trade_style.upper() == "SCALP")

        if scalp_mode:
            # Pure Sniper: zero tolerance for traps or structure breaks
            min_force = 1.2
            must_resonate = True
            must_struct = True
            vol_req = 1.0
        else:
            # Swing Sniper: safer but allows more early signals
            min_force = 0.8
            must_resonate = False
            must_struct = False
            vol_req = 0.9

        # =======================================================================
        # 7. Signal Logic
        # =======================================================================
        bias = "HOLD"
        lane = "⚫ HOLD"
        score = 50
        vol_scalar = float(mass.iloc[-1])

        # ------------------------ LONG LOGIC ------------------------
        long_ok = (
            force_now > min_force
            and vol_scalar > vol_req
            and not trap
            and (not must_struct or bull_struct)
            and (not must_resonate or resonance)
        )

        # ------------------------ SHORT LOGIC ------------------------
        short_ok = (
            force_now < -min_force
            and vol_scalar > vol_req
            and not trap
            and (not must_struct or bear_struct)
            and (not must_resonate or resonance)
        )

        # ---------------------- Mode-based logic --------------------
        if regime == "EXPANSION":
            if long_ok:
                bias, lane, score = "LONG", "🚀 BREAKOUT", 90
            elif short_ok:
                bias, lane, score = "SHORT", "🚀 BREAKOUT", 90

        elif regime == "TREND":
            if long_ok:
                bias, lane, score = "LONG", "🟢 SNIPER", 85
            elif short_ok:
                bias, lane, score = "SHORT", "🟢 SNIPER", 85

        elif regime == "COMPRESSION":
            score = 65
            bias = "WATCH"
            lane = "🟠 BUILDING"

        elif regime == "EXHAUSTION":
            score = 45
            bias = "HOLD"
            lane = "🔴 EXHAUSTION"

        # =======================================================================
        # 8. Adaptive Targeting + Risk
        # =======================================================================
        sigma_now = float(sigma.iloc[-1])
        stop = t1 = t2 = t3 = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1

            if scalp_mode:
                stop_mult = 0.9
                t_mult = (1.2, 2.0, 3.0)
                base_risk = self.BASE_RISK_SCALP
            else:
                stop_mult = 1.4
                t_mult = (2.0, 4.0, 7.0)
                base_risk = self.BASE_RISK_SWING

            stop = price - direction * sigma_now * stop_mult
            t1 = price + direction * sigma_now * t_mult[0]
            t2 = price + direction * sigma_now * t_mult[1]
            t3 = price + direction * sigma_now * t_mult[2]

            rr = abs(t1 - price) / abs(price - stop)
            risk_pct = base_risk * min(1.5, abs(force_now))
        else:
            rr = 0
            risk_pct = 0

        # =======================================================================
        # 9. Premium UI Features
        # =======================================================================
        whale_z = float(vol_scalar - 1.0)
        whale_state = "ACTIVE" if abs(whale_z) >= 2.0 else "BASELINE"
        whale_label = "Institutional" if whale_state == "ACTIVE" else "Normal"

        top_features = []

        # LONG / SHORT descriptions
        if bias in ["LONG", "SHORT"]:
            if lane == "🚀 BREAKOUT":
                top_features.append({"desc": "High Momentum Breakout", "importance": 95})
            else:
                desc = "Strong Bullish Momentum" if bias == "LONG" else "Strong Bearish Momentum"
                top_features.append({"desc": desc, "importance": 90})

            if resonance:
                top_features.append({"desc": "Multi-TF Confluence", "importance": 85})

            if whale_state == "ACTIVE":
                top_features.append({"desc": "Institutional Volume Spike", "importance": 80})
            else:
                top_features.append({"desc": "Healthy Volume Profile", "importance": 70})

        elif bias == "WATCH":
            top_features.append({"desc": "Volatility Compression", "importance": 80})
            top_features.append({"desc": "Awaiting Breakout", "importance": 60})

        else:
            top_features.append({"desc": "No Clear Edge", "importance": 40})

        top_features = top_features[:3]

        narrative = self._narrative(regime, bias, whale_state, resonance)

        return SimpleNamespace(
            bias=bias,
            lane=lane,
            score=score,
            price=price,
            entry=price if bias != "HOLD" else 0.0,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=round(rr, 2),
            risk_pct=round(risk_pct * 100, 2),
            regime=regime,
            regime_color="green" if bias != "HOLD" else "gray",
            whale_state=whale_state,
            whale_label=whale_label,
            top_features=top_features,
            narrative=narrative,
            lifecycle="ACTIVE" if bias != "HOLD" else "WAITING"
        )

    # ==========================================================================
    # HELPERS
    # ==========================================================================
    def _narrative(self, regime, bias, whale, resonance):
        if bias == "HOLD":
            if regime == "COMPRESSION":
                return "Market compressing, energy building."
            if regime == "EXHAUSTION":
                return "Momentum cooling."
            return "No actionable structure."

        if regime == "EXPANSION":
            return "Volatility breakout confirmed."
        if resonance:
            return "Strong multi-TF alignment."
        if whale == "ACTIVE":
            return "Institutional volume active."
        return "Momentum and structure aligned."

    def _neutral(self, price, reason):
        return SimpleNamespace(
            bias="HOLD",
            lane="⚫ HOLD",
            score=50,
            price=price,
            entry=0, stop=0, target1=0, target2=0, target3=0,
            rr_ratio=0, risk_pct=0,
            regime="NEUTRAL",
            regime_color="gray",
            whale_state="BASELINE",
            whale_label="Normal",
            narrative=reason,
            top_features=[{"desc": "Initializing...", "importance": 10}],
            lifecycle="WAITING"
        )
