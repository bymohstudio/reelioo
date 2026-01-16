import numpy as np
import pandas as pd
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    BACKTEST ENGINE v28 – INSTITUTIONAL GRADE

    - Perfectly mirrors CryptoQuantEngine v28 logic.
    - Includes: Vector Smoothing, EMA 50 Governor, Strict Thresholds.
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.trades = []

        # v28 Constants
        self.SIGMA_WINDOW = 14
        self.MASS_WINDOW = 20
        self.STRUCT_WINDOW = 20
        self.FEE = 0.06  # Binance Taker %

    def run(self, mode="INTRADAY"):
        if len(self.df) < 60:
            return self._empty_result()

        # 1. Feature Generation (Manual calculation to ensure match)
        df = self.df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ==================================================
        # 2. PHYSICS CONSTRUCTION (v28 LOGIC)
        # ==================================================

        # A. Sigma (Volatility)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        sigma = tr.rolling(self.SIGMA_WINDOW).mean()

        # B. Kinetic Vectors
        velocity = close.diff() / (sigma + 1e-9)
        mass_mean = volume.rolling(self.MASS_WINDOW).mean()
        mass = volume / (mass_mean + 1e-9)

        raw_force = mass * velocity

        # [v28] Hysteresis Smoothing
        smooth_force = raw_force.rolling(3).mean()
        force_decay = smooth_force.diff(3)

        # C. Equilibrium Governor (EMA 50)
        equilibrium = close.ewm(span=50, adjust=False).mean()

        # D. Structure
        roll_high = high.rolling(self.STRUCT_WINDOW).max().shift(1)
        roll_low = low.rolling(self.STRUCT_WINDOW).min().shift(1)

        # E. Anomaly Filters
        candle_range = high - low
        is_wide = candle_range > (2.5 * sigma)
        mass_intensity = volume / (mass_mean + 1e-9)
        is_hollow = is_wide & (mass_intensity < 0.9)

        # ------------------------------------------------
        # 3. SIMULATION LOOP
        # ------------------------------------------------
        position = None
        entry = stop = target = 0.0
        direction = 0

        # Start after smoothing window is valid
        start_idx = 55

        for i in range(start_idx, len(df)):

            # Context
            price = close.iloc[i]
            c_low = low.iloc[i]
            c_high = high.iloc[i]
            c_sigma = sigma.iloc[i]

            # Physics
            force_now = smooth_force.iloc[i]
            decay_val = force_decay.iloc[i]
            c_mass = mass.iloc[i]

            # Structure
            base_line = equilibrium.iloc[i]
            r_high = roll_high.iloc[i]
            r_low = roll_low.iloc[i]

            # Trap Check
            is_trap = is_hollow.iloc[i] or (abs(force_now) > 3.5)

            # --------------------------------------------
            # EXIT LOGIC
            # --------------------------------------------
            if position:
                exit_res = None
                exit_price = price

                if direction == 1:  # LONG
                    if c_low <= stop:
                        exit_res = "STOP"
                        exit_price = stop
                    elif c_high >= target:
                        exit_res = "WIN"
                        exit_price = target
                else:  # SHORT
                    if c_high >= stop:
                        exit_res = "STOP"
                        exit_price = stop
                    elif c_low <= target:
                        exit_res = "WIN"
                        exit_price = target

                if exit_res:
                    pnl = (exit_price - entry) / entry * 100
                    if direction == -1: pnl *= -1
                    pnl -= self.FEE  # Fee impact

                    self.trades.append({
                        "result": exit_res,
                        "pnl": round(pnl, 2),
                        "entry": entry,
                        "exit": exit_price,
                    })
                    position = None
                    continue

            # --------------------------------------------
            # ENTRY LOGIC (v28 Strict)
            # --------------------------------------------
            if not position:

                # 1. Determine Regime (v28 Thresholds)
                regime = "IDLE"
                if abs(force_now) < 0.6:
                    regime = "COMPRESSION"
                elif abs(force_now) > 2.0 and decay_val > 0:
                    regime = "EXPANSION"
                elif abs(force_now) > 0.8:
                    regime = "TREND"
                elif decay_val < -0.5:
                    regime = "EXHAUSTION"

                bias = "HOLD"
                lane = "HOLD"

                # Quality Check
                is_high_quality = c_mass > 1.2

                # Governor Check
                is_positive = price > base_line
                is_negative = price < base_line

                # --- A. TREND ---
                if regime == "TREND":
                    if force_now > 0 and is_positive and not is_trap and is_high_quality:
                        bias = "LONG"
                        lane = "TREND"
                    elif force_now < 0 and is_negative and not is_trap and is_high_quality:
                        bias = "SHORT"
                        lane = "TREND"

                # --- B. EXPANSION ---
                elif regime == "EXPANSION":
                    if force_now > 2.0 and not is_trap and is_positive:
                        bias = "LONG"
                        lane = "BREAKOUT"
                    elif force_now < -2.0 and not is_trap and is_negative:
                        bias = "SHORT"
                        lane = "BREAKOUT"

                # --- C. RANGE (Compression) ---
                elif regime == "COMPRESSION":
                    # Range Width Check (>1.5%)
                    if r_low > 0:
                        width = (r_high - r_low) / r_low
                        if width > 0.015:
                            dist_sup = (price - r_low) / r_low
                            dist_res = (r_high - price) / r_high
                            c_vel = velocity.iloc[i]

                            # Long Support (Only if NOT crashing)
                            if dist_sup < 0.01 and c_vel > 0 and not is_negative:
                                bias = "LONG"
                                lane = "RANGE"

                            # Short Resistance (Only if NOT pumping)
                            elif dist_res < 0.01 and c_vel < 0 and not is_positive:
                                bias = "SHORT"
                                lane = "RANGE"

                # 3. Execution
                if bias != "HOLD":
                    direction = 1 if bias == "LONG" else -1

                    # Sizing based on Lane
                    if lane == "RANGE":
                        stop_mult = 1.0
                        target_mult = 1.0  # 1:1 RR for Range
                    elif lane == "BREAKOUT":
                        stop_mult = 1.8
                        target_mult = 3.0
                    else:  # Trend
                        stop_mult = 1.5
                        target_mult = 3.5

                    entry = price
                    stop = price - (direction * c_sigma * stop_mult)

                    # For Range, Target is Fixed Level
                    if lane == "RANGE":
                        target = r_high if bias == "LONG" else r_low
                    else:
                        target = price + (direction * c_sigma * target_mult)

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
            "recent_trades": self.trades[-20:]
        }

    def _empty_result(self):
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "recent_trades": []}