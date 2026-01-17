from __future__ import annotations
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    BACKTEST ENGINE v31 – QUANTUM TITANIUM

    PERFECTLY MIRRORS v31 ENGINE LOGIC:
    -----------------------------------
    ✓ Anti-Lag Force (2-period smoothing)
    ✓ Acceleration Gating (Don't buy dying moves)
    ✓ Elasticity Limits (Don't buy tops)
    ✓ Energy Reserves (Don't short floors)
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.trades = []

        # === v31 Constants ===
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 50
        self.MAX_STRETCH = 0.03  # 3% limit
        self.FEE = 0.06  # % taker fee

    def run(self, mode: str = "INTRADAY"):
        if len(self.df) < 80:
            return self._empty_result()

        df = self.df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        open_p = df["open"]

        # ==================================================
        # 1. QUANTUM VECTORS (MATCH v31)
        # ==================================================

        # Volatility
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        sigma = tr.rolling(self.ATR_LEN).mean()

        # Momentum & Mass
        velocity = close.diff() / (sigma + 1e-9)
        vol_mean = volume.rolling(self.MASS_LEN).mean()
        mass = volume / (vol_mean + 1e-9)

        # Force (Anti-Lag Smoothing)
        force = (mass * velocity).rolling(2).mean()

        # Acceleration (Jerk) - CRITICAL for v31
        acceleration = force.diff(2)

        # ==================================================
        # 2. ENERGY & ELASTICITY
        # ==================================================

        # Energy (Hidden RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        energy_reserve = 100 - (100 / (1 + rs))

        # Elasticity (Baseline Stretch)
        eq = close.ewm(span=self.STRUCT_LEN).mean()
        stretch_pct = (close - eq) / eq

        is_overstretched_long = stretch_pct > self.MAX_STRETCH
        is_overstretched_short = stretch_pct < -self.MAX_STRETCH

        # Structure
        bull_struct = close > eq
        bear_struct = close < eq

        # ==================================================
        # 3. TRAP DETECTION
        # ==================================================
        body = (close - open_p).abs()
        upper_wick = high - pd.concat([close, open_p], axis=1).max(axis=1)
        lower_wick = pd.concat([close, open_p], axis=1).min(axis=1) - low

        # ==================================================
        # 4. SIMULATION LOOP
        # ==================================================
        position = None
        entry = stop = target = 0.0
        direction = 0

        start_idx = 60

        for i in range(start_idx, len(df)):

            # Current Slice
            price = float(close.iloc[i])
            c_low = float(low.iloc[i])
            c_high = float(high.iloc[i])
            c_sigma = float(sigma.iloc[i])

            # Physics
            f_now = float(force.iloc[i])
            acc_now = float(acceleration.iloc[i])

            # Context
            en_now = float(energy_reserve.iloc[i])
            w_up = float(upper_wick.iloc[i])
            w_down = float(lower_wick.iloc[i])
            c_body = float(body.iloc[i])

            # Trap Checks
            is_wick_trap = False
            if f_now > 0 and w_up > (c_body * 1.2): is_wick_trap = True
            if f_now < 0 and w_down > (c_body * 1.2): is_wick_trap = True

            is_exhaustion = (abs(f_now) > 1.5 and acc_now < 0)

            # Regime
            if abs(f_now) < 0.6:
                regime = "COMPRESSION"
            elif abs(f_now) > 2.0 and acc_now > 0:
                regime = "EXPANSION"
            elif abs(f_now) > 0.8:
                regime = "TREND"
            else:
                regime = "IDLE"

            # ----------------------------------------------
            # EXIT LOGIC
            # ----------------------------------------------
            if position:
                exit_price = price
                result = None

                if direction == 1:  # Long Exit
                    if c_low <= stop:
                        exit_price = stop
                        result = "LOSS"
                    elif c_high >= target:
                        exit_price = target
                        result = "WIN"
                else:  # Short Exit
                    if c_high >= stop:
                        exit_price = stop
                        result = "LOSS"
                    elif c_low <= target:
                        exit_price = target
                        result = "WIN"

                if result:
                    pnl = (exit_price - entry) / entry * 100
                    if direction == -1: pnl *= -1
                    pnl -= self.FEE

                    self.trades.append({
                        "result": result,
                        "pnl": round(pnl, 2),
                        "entry": entry,
                        "exit": exit_price,
                    })
                    position = None
                    continue

            # ----------------------------------------------
            # ENTRY LOGIC (UPDATED TO MATCH v32.1 STRICT MODE)
            # ----------------------------------------------
            if not position and regime in ["TREND", "EXPANSION"]:

                bias = None

                # Filters
                valid_long_energy = en_now < 75
                valid_short_energy = en_now > 25
                vol_ok = float(mass.iloc[i]) > 1.0

                # --- v32.1 CONSTANTS ---
                MIN_FORCE = 1.2  # Was 0.8
                MIN_ACCEL = 0.05  # Was -0.1
                STOP_MULT = 1.2  # Was 1.5 (Tighter stops for impulse)

                # LONG
                if (f_now > MIN_FORCE and
                        acc_now > MIN_ACCEL and
                        bull_struct.iloc[i] and
                        not is_overstretched_long.iloc[i] and
                        valid_long_energy and
                        vol_ok and
                        not is_wick_trap and
                        not is_exhaustion):
                    bias = "LONG"

                # SHORT
                elif (f_now < -MIN_FORCE and
                      acc_now < -MIN_ACCEL and
                      bear_struct.iloc[i] and
                      not is_overstretched_short.iloc[i] and
                      valid_short_energy and
                      vol_ok and
                      not is_wick_trap and
                      not is_exhaustion):
                    bias = "SHORT"

                # EXECUTE
                if bias:
                    direction = 1 if bias == "LONG" else -1
                    entry = price

                    # Updated Stop Distance for v32.1
                    stop_dist = c_sigma * STOP_MULT
                    stop = price - (direction * stop_dist)

                    # Target 2.0
                    target = price + (direction * stop_dist * 2.0)

                    position = bias

        return self._stats()

    def _stats(self):
        if not self.trades: return self._empty_result()
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
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "recent_trades": []}