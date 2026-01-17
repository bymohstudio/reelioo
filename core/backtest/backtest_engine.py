from __future__ import annotations
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    BACKTEST ENGINE v30 – TITANIUM GRADE

    Mirrors:
    - Signed kinetic energy
    - Energy decay (relief / exhaustion)
    - Structural dominance
    - Trap rejection
    - Mode-aware logic (SWING vs SCALP)

    NO UI / NO CRON / PURE SIMULATION
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.trades = []

        # === Physics constants (must match engine) ===
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20
        self.FEE = 0.06  # % taker fee

    # ==========================================================
    # RUN
    # ==========================================================

    def run(self, mode: str = "INTRADAY"):
        if len(self.df) < 80:
            return self._empty_result()

        df = self.df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ==================================================
        # 1. TRUE PHYSICS (MATCH v30)
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

        signed_energy = mass * velocity
        energy_decay = signed_energy.diff(3)

        # ==================================================
        # 2. STRUCTURE (NO EMA)
        # ==================================================

        hh = high > high.rolling(self.STRUCT_LEN).max().shift(1)
        hl = low > low.rolling(self.STRUCT_LEN).min().shift(1)

        lh = high < high.rolling(self.STRUCT_LEN).max().shift(1)
        ll = low < low.rolling(self.STRUCT_LEN).min().shift(1)

        # ==================================================
        # 3. TRAP / FAKE MOVE DETECTION
        # ==================================================

        candle_range = (high - low) / close
        vol_intensity = volume / (vol_mean + 1e-9)

        is_fake = (candle_range > 0.02) & (vol_intensity < 0.8)

        # ==================================================
        # 4. SIMULATION LOOP
        # ==================================================

        position = None
        entry = stop = target = 0.0
        direction = 0

        start_idx = 60

        for i in range(start_idx, len(df)):

            price = close.iloc[i]
            c_low = low.iloc[i]
            c_high = high.iloc[i]
            c_atr = atr.iloc[i]

            ke = signed_energy.iloc[i]
            decay = energy_decay.iloc[i]

            # Structure state
            struct_up = hh.iloc[i] and hl.iloc[i]
            struct_down = lh.iloc[i] and ll.iloc[i]

            fake = is_fake.iloc[i]

            # ==================================================
            # EXIT LOGIC
            # ==================================================
            if position:
                exit_price = price
                result = None

                if direction == 1:
                    if c_low <= stop:
                        exit_price = stop
                        result = "LOSS"
                    elif c_high >= target:
                        exit_price = target
                        result = "WIN"
                else:
                    if c_high >= stop:
                        exit_price = stop
                        result = "LOSS"
                    elif c_low <= target:
                        exit_price = target
                        result = "WIN"

                if result:
                    pnl = (exit_price - entry) / entry * 100
                    if direction == -1:
                        pnl *= -1
                    pnl -= self.FEE

                    self.trades.append({
                        "result": result,
                        "pnl": round(pnl, 2),
                        "entry": entry,
                        "exit": exit_price,
                    })

                    position = None
                    continue

            # ==================================================
            # ENTRY LOGIC (v30)
            # ==================================================

            if position:
                continue

            # ---- REGIME DETECTION ----
            if abs(ke) < 0.5:
                regime = "COMPRESSION"
            elif abs(ke) > 1.5 and decay > 0:
                regime = "EXPANSION"
            elif abs(ke) > 0.8:
                regime = "TREND"
            elif decay < 0:
                regime = "EXHAUSTION"
            else:
                regime = "IDLE"

            bias = None

            # ---- MODE FILTERS ----
            if mode == "SCALP":
                ke_req = 0.6
                rr_mult = 1.8
                stop_mult = 1.2
            else:  # INTRADAY = SWING
                ke_req = 1.0
                rr_mult = 3.0
                stop_mult = 1.5

            # ---- SIGNAL CONDITIONS ----
            if regime in ("TREND", "EXPANSION") and not fake:

                # LONG
                if ke > ke_req and struct_up and decay >= 0:
                    bias = "LONG"

                # SHORT (RELIEF CONFIRMATION)
                elif ke < -ke_req and struct_down and decay < 0:
                    bias = "SHORT"

            # ---- EXECUTION ----
            if bias:
                direction = 1 if bias == "LONG" else -1
                entry = price

                stop = price - direction * c_atr * stop_mult
                target = price + direction * c_atr * rr_mult

                position = bias

        return self._stats()

    # ==========================================================
    # STATS
    # ==========================================================

    def _stats(self):
        if not self.trades:
            return self._empty_result()

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] < 0]

        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))

        pf = round(gross_profit / gross_loss, 2) if gross_loss else 10.0

        return {
            "total_trades": len(self.trades),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1),
            "profit_factor": pf,
            "recent_trades": self.trades[-20:],
        }

    def _empty_result(self):
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "recent_trades": [],
        }
