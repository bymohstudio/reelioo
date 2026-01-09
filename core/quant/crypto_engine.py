# core/quant/crypto_engine.py
from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoQuantEngine:
    """
    REELIOO PHYSICS ENGINE v18.0 (KINETIC POTENTIAL)

    Backend: Uses high-fidelity physics (Kinetic/Potential Energy).
    Frontend: Outputs punchy, "Retail Friendly" signal tags.
    """

    def __init__(self):
        # Physics Constants
        self.MASS_WINDOW = 20
        self.VELOCITY_WINDOW = 3
        self.GRAVITY = 9.8

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0
        try:
            # --- 1. CONSTRUCT THE PHYSICS WORLD ---
            close = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']
            price = float(close.iloc[-1])

            # A. MASS (m)
            vol_avg = volume.rolling(self.MASS_WINDOW).mean()
            mass = volume / (vol_avg + 0.0001)

            # B. VELOCITY (v)
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            velocity = close.diff(self.VELOCITY_WINDOW) / (atr + 0.0001)

            # C. ACCELERATION (a)
            acceleration = velocity.diff()

            # D. FORCE (F = ma)
            force = mass * acceleration

            # E. ENERGIES
            # Kinetic (KE): Active Momentum
            ke = 0.5 * mass * (velocity ** 2)

            # Potential (PE): Compression
            normalized_volatility = atr / close
            pe = (1.0 / (normalized_volatility + 0.001)).rolling(10).mean()
            pe_score = (pe - pe.rolling(50).min()) / (pe.rolling(50).max() - pe.rolling(50).min() + 0.001) * 100

            # --- SNAPSHOT ---
            c_mass = float(mass.iloc[-1])
            c_velo = float(velocity.iloc[-1])
            c_force = float(force.iloc[-1])
            c_ke = float(ke.iloc[-1])
            c_pe = float(pe_score.iloc[-1])
            c_atr = float(atr.iloc[-1])
            baseline = close.ewm(span=50).mean().iloc[-1]

        except Exception as e:
            return self._neutral_result(0.0, f"Data Error: {e}")

        # ==========================================
        # PHASE 1: IDENTIFY THE STATE
        # ==========================================
        is_compressed = c_pe > 80 and c_ke < 1.0
        is_exploding = c_ke > 2.0 and c_mass > 1.2
        is_trending = c_ke > 1.0 and abs(c_velo) > 0.5

        # ==========================================
        # PHASE 2: CALCULATE DRIVERS
        # ==========================================
        score = 50
        drivers = []
        bias = "HOLD"

        # VECTOR 1: FORCE (MOMENTUM)
        if c_force > 0 and c_velo > 0:
            score += 15
            drivers.append({"desc": "Bullish Momentum", "importance": 80})
        elif c_force < 0 and c_velo < 0:
            score += 15
            drivers.append({"desc": "Bearish Pressure", "importance": 80})

        # VECTOR 2: MASS (VOLUME)
        if c_mass > 1.5:
            score += 15
            drivers.append({"desc": "High Volume", "importance": 90})
        elif c_mass < 0.5:
            score -= 10

        # VECTOR 3: KINETIC (TREND)
        if is_exploding:
            score += 25
            drivers.append({"desc": "Explosive Breakout", "importance": 95})
        elif is_trending:
            score += 10
            drivers.append({"desc": "Trend Continuation", "importance": 70})

        # VECTOR 4: POTENTIAL (SQUEEZE)
        if is_compressed:
            drivers.append({"desc": "Volatility Squeeze", "importance": 85})
            if c_velo > 0: score += 5
            if c_velo < 0: score += 5

        # ==========================================
        # PHASE 3: DIRECTION
        # ==========================================
        if c_velo > 0 and price > baseline:
            bias = "LONG"
        elif c_velo < 0 and price < baseline:
            bias = "SHORT"
        else:
            bias = "WATCH"
            score = 50

        score = min(99, max(1, score))

        # Lane Logic
        if score >= 80:
            lane = f"🟢 POWER {bias}"
        elif score >= 60:
            lane = f"🟡 {bias}"
        elif is_compressed:
            lane = "🟠 CHARGING"
            bias = "WATCH"
            score = 60
        else:
            lane = "⚫ IDLE"
            bias = "HOLD"
            score = 40

        # ==========================================
        # PHASE 4: TARGETING
        # ==========================================
        entry = price
        stop = 0.0
        t1 = t2 = t3 = 0.0

        if bias in ["LONG", "SHORT"]:
            direction = 1 if bias == "LONG" else -1
            stop_dist = c_atr * 1.5
            stop = entry - (direction * stop_dist)
            throw_power = max(1.0, min(3.0, c_ke))
            t1 = entry + (direction * c_atr * 2.0)
            t2 = entry + (direction * c_atr * 3.0 * throw_power)
            t3 = entry + (direction * c_atr * 5.0 * throw_power)

        # Generate Punchy Narrative
        narrative = self._build_narrative(bias, c_ke, c_pe, c_mass, c_force)

        # Regime Color
        regime_color = "gray"
        if c_ke > 2.0: regime_color = "green"
        if is_compressed: regime_color = "blue"

        return SimpleNamespace(
            bias=bias, lane=lane, score=score, price=price,
            entry=entry, stop=round(stop, 4), target1=round(t1, 4), target2=round(t2, 4), target3=round(t3, 4),
            rr_ratio=round(abs(t2 - entry) / abs(entry - stop), 2) if stop != 0 else 0.0,
            expected_duration="8h",
            regime="HIGH MOMENTUM" if c_ke > 1.5 else "SQUEEZE" if is_compressed else "LOW VOLATILITY",
            regime_color=regime_color,
            whale_label="High" if c_mass > 1.5 else "Normal",
            top_features=drivers[:3],
            narrative=narrative,
            lifecycle="ACTIVE" if bias in ["LONG", "SHORT"] else "WAITING"
        )

    def _build_narrative(self, bias, ke, pe, mass, force):
        """
        Returns 1-3 word punchy tags for the UI.
        """
        if bias == "HOLD": return "NO SIGNAL"

        # 1. High Energy States
        if pe > 80: return "VOLATILITY SQUEEZE"
        if ke > 2.0: return "EXPLOSIVE BREAKOUT"

        # 2. Institutional Flows
        if mass > 1.5:
            return "INSTITUTIONAL BUYING" if bias == "LONG" else "INSTITUTIONAL DUMP"

        # 3. Momentum States
        if abs(force) > 2.0:
            return "MOMENTUM SURGE" if bias == "LONG" else "HEAVY SELLING"

        # 4. Standard Trend
        if bias == "LONG": return "BULLISH TREND"
        if bias == "SHORT": return "BEARISH TREND"
        if bias == "WATCH": return "STRUCTURE FORMING"

        return "ANALYZING..."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD", lane="⚫ HOLD", score=50, price=price, entry=0.0,
            stop=0.0, target1=0.0, target2=0.0, target3=0.0,
            rr_ratio=0.0, expected_duration="--",
            regime="CALIBRATING", regime_color="gray",
            whale_label="--", top_features=[], narrative=reason, lifecycle="WAITING"
        )