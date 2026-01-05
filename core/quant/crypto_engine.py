from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
import logging

from core.quant.feature_engineering import generate_features
from core.quant.regime_memory import apply_regime_memory
from core.quant.btc_regime import update_btc_regime, apply_btc_gating

log = logging.getLogger(__name__)


def cap(x, limit=3.0):
    return max(-limit, min(limit, x))


class CryptoQuantEngine:
    """
    REELIOO QUANT PHYSICS ENGINE v8.7 (Institutional Grade)

    - Physics Alpha (unchanged)
    - Institutional Gates (unchanged)
    - Regime Memory (stateful confidence decay)
    - BTC-First Global Risk Gating
    """

    def __init__(self):
        self.SIGMOID_K = 0.45
        self.CONFIRMATION_THRESH = 70
        self.MIN_VOLUME_RATIO = 0.6
        self.MAX_ATR_PCT = 0.04

        log.info("🚀 QuantPhysicsEngine v8.7 Initialized")

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def analyze(self, df: pd.DataFrame, trade_style: str = "DAY") -> SimpleNamespace:
        price = 0.0

        try:
            # =========================
            # DATA ENRICHMENT
            # =========================
            df = generate_features(df)
            if df.empty:
                return self._neutral_result(0.0, "No Data")

            if "live_close" in df.columns and not pd.isna(df.iloc[-1]["live_close"]):
                price = float(df.iloc[-1]["live_close"])
            else:
                price = float(df.iloc[-1]["close"])

            mass = df.get("quote_volume", df["volume"])
            velocity = df["close"].diff()
            trades = df.get("trades", 1)

            df["force"] = mass * velocity
            df["friction_coeff"] = trades / (mass + 1)

            last = df.iloc[-1]

        except Exception as e:
            return self._neutral_result(price, f"Data Error: {e}")

        # =========================
        # PHASE 1 — PHYSICS ALPHA
        # =========================
        avg_friction = df["friction_coeff"].rolling(20).mean().iloc[-1]
        current_friction = df["friction_coeff"].iloc[-1]
        is_stagnant = (
            current_friction > avg_friction * 1.5
            and abs(last.get("ret_1", 0)) < 0.001
        )

        trend_alpha = cap(last.get("ema_diff", 0) * 100) * 1.5
        whale_z = cap(float(last.get("whale_z", 0)))
        reversion_alpha = -cap(last.get("vwap_dist", 0) * 100) * 1.2

        kinetic = cap(float(last.get("kinetic_energy", 0)))
        shock = cap(float(last.get("momentum_shock", 0)) * 5)
        compression = float(last.get("volatility_compression", 1.0))

        physics_alpha = (kinetic * 0.8) + shock
        is_spring_loaded = compression < 0.6 and abs(whale_z) > 0.8
        if is_spring_loaded:
            physics_alpha += 3.0 * np.sign(trend_alpha + shock)

        raw_alpha = trend_alpha + whale_z + reversion_alpha + physics_alpha
        final_probability = self._sigmoid(raw_alpha)

        # =========================
        # PHASE 2 — GATEKEEPERS
        # =========================
        session_mod = 0
        try:
            hour = last.name.hour
            if 0 <= hour < 8:
                session_mod = -5
            elif 13 <= hour < 16:
                session_mod = +5
        except Exception:
            pass

        adjusted_score = final_probability + session_mod

        execution_gate = True
        gate_reason = ""

        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        if last["volume"] < avg_vol * self.MIN_VOLUME_RATIO:
            execution_gate = False
            gate_reason = "Low Liquidity"

        if float(last.get("atr_pct", 0)) > self.MAX_ATR_PCT:
            execution_gate = False
            gate_reason = "Volatility Shock"

        kill_switch = abs(shock) > 2.8
        if kill_switch:
            gate_reason = "Extreme Volatility"

        # =========================
        # PHASE 3 — REGIME MEMORY (CORRECT PLACE)
        # =========================
        if is_stagnant or not execution_gate or kill_switch:
            adjusted_score = 50

        adjusted_score = apply_regime_memory(
            symbol=str(df.index.name or "UNKNOWN"),
            bias="RAW",
            score=int(adjusted_score),
        )

        # =========================
        # PHASE 4 — FINAL BIAS
        # =========================
        # -------------------------
        # FINAL SCORE (DISPLAY SCORE)
        # -------------------------
        score = int(adjusted_score)
        score = 50 + (score - 50) * 0.95
        score = int(round(score))

        # -------------------------
        # FINAL BIAS (DERIVED FROM DISPLAY SCORE)
        # -------------------------
        bias = "HOLD"

        if score >= 70:
            bias = "LONG"
        elif score <= 30:
            bias = "SHORT"
            score = 100 - score
        elif 60 <= score < 70:
            bias = "WATCH"
        elif 30 < score <= 40:
            bias = "WATCH"
            score = 100 - score
        else:
            bias = "HOLD"
            score = 50

        # =========================
        # BTC-FIRST GLOBAL GATING
        # =========================
        symbol_name = str(df.index.name or "UNKNOWN")

        if symbol_name.startswith("BTC"):
            update_btc_regime(bias, int(score))
        else:
            bias, score = apply_btc_gating(symbol_name, bias, int(score))

        # =========================
        # PHASE 5 — OUTPUT
        # =========================
        lifecycle = "WAITING"
        if bias == "WATCH":
            lifecycle = "EMERGING"
        if bias in ["LONG", "SHORT"]:
            lifecycle = "CONFIRMED"
        if kill_switch:
            lifecycle = "SHIELDED"

        entry = stop = t1 = t2 = t3 = 0.0
        rr_ratio = 0.0
        expected_duration = "--"

        if bias in ["LONG", "SHORT"]:
            entry = price
            atr = float(last.get("atr_14", price * 0.01))
            direction = 1 if bias == "LONG" else -1
            extension = 1.0 + abs(physics_alpha) * 0.2

            stop = price - direction * atr * 1.5
            t1 = price + direction * atr * 2.0 * extension
            t2 = price + direction * atr * 3.5 * extension
            t3 = price + direction * atr * 5.0 * extension

            risk = abs(entry - stop)
            if risk > 0:
                rr_ratio = round(abs(t1 - entry) / risk, 2)
            expected_duration = "4h"

        return SimpleNamespace(
            bias=bias,
            score=int(score),
            price=price,
            entry=entry,
            stop=round(stop, 4),
            target1=round(t1, 4),
            target2=round(t2, 4),
            target3=round(t3, 4),
            rr_ratio=rr_ratio,
            expected_duration=expected_duration,
            regime="IMPULSE" if abs(physics_alpha) > 1.5 else "FLOW",
            regime_color="green" if bias == "LONG" else "red" if bias == "SHORT" else "yellow" if bias == "WATCH" else "gray",
            whale_zscore=round(whale_z, 2),
            whale_label="High" if abs(whale_z) > 1.5 else "Normal",
            top_features=[],
            narrative=self._build_narrative(bias, score, is_stagnant, is_spring_loaded, kill_switch, gate_reason),
            lifecycle=lifecycle,
            flow_score=0.5,
        )

    def _build_narrative(self, bias, score, is_stagnant, is_spring, kill_switch, gate_reason):
        if kill_switch:
            return f"🛡️ RISK PROTOCOL ACTIVE: {gate_reason}"
        if is_stagnant:
            return "⚠️ Retail Trap Detected. Market chopping."
        if bias == "HOLD":
            return "System Idle. No Edge."
        if bias == "WATCH":
            return f"Momentum building ({score}%)."
        if is_spring:
            return "⚡ SQUEEZE SETUP: Volatility compressed."

        return "Institutional momentum confirmed."

    def _neutral_result(self, price, reason):
        return SimpleNamespace(
            bias="HOLD",
            score=50,
            price=price,
            entry=0.0,
            stop=0.0,
            target1=0.0,
            target2=0.0,
            target3=0.0,
            rr_ratio=0,
            expected_duration="--",
            regime="WAIT",
            regime_color="gray",
            whale_zscore=0,
            whale_label="Normal",
            top_features=[],
            narrative=reason,
            lifecycle="WAITING",
            flow_score=0.5,
        )
