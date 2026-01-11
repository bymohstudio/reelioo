import numpy as np
import pandas as pd
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    BACKTEST ENGINE v21 – TRUE PHYSICS ALIGNMENT

    - Perfectly mirrors CryptoQuantEngine v21 logic.
    - Uses Signed Kinetic Energy, Decay, and Regimes.
    - Dynamic ATR-based fakeout detection.
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.trades = []

        # v21 Constants
        self.ATR_LEN = 14
        self.MASS_LEN = 20
        self.STRUCT_LEN = 20
        self.FEE = 0.06  # Binance Taker % (approx)

    # ------------------------------------------------
    # CORE RUN
    # ------------------------------------------------
    def run(self, mode="INTRADAY"):
        if self.df.empty:
            return self._empty_result()

        # 1. Generate Features
        try:
            df = generate_features(self.df)
        except Exception as e:
            log.error(f"Backtest Feature Error: {e}")
            return self._empty_result()

        # ------------------------------------------------
        # 2. PHYSICS CONSTRUCTION (Match v21)
        # ------------------------------------------------
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- A. PHYSICS VECTORS ---
        # True Range & ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.ATR_LEN).mean()

        # Velocity & Mass
        velocity = close.diff() / (atr + 1e-9)
        vol_mean = volume.rolling(self.MASS_LEN).mean()
        mass = volume / (vol_mean + 1e-9)

        # Kinetic Energy & Decay
        signed_ke = mass * velocity
        ke_decay = signed_ke.diff(3)

        # --- B. STRUCTURE (Lagged) ---
        roll_high = high.rolling(self.STRUCT_LEN).max().shift(1)
        roll_low = low.rolling(self.STRUCT_LEN).min().shift(1)

        hh = high > roll_high
        hl = low > roll_low
        lh = high < roll_high
        ll = low < roll_low

        structure_up = hh & hl
        structure_down = lh & ll

        # --- C. FAKEOUT DETECTION (Dynamic) ---
        # v21 Logic: Range > 2.5 ATR and Low Volume Intensity
        candle_range = high - low
        is_wide = candle_range > (2.5 * atr)
        vol_intensity = volume / (vol_mean + 1e-9)
        is_fake = is_wide & (vol_intensity < 0.8)

        # --- D. RESONANCE ---
        ht_velocity = close.diff(5) / (atr + 1e-9)
        ht_ke = (mass * ht_velocity).rolling(5).mean()
        resonance = np.sign(signed_ke) == np.sign(ht_ke)

        # ------------------------------------------------
        # 3. SIMULATION LOOP
        # ------------------------------------------------
        position = None
        entry = stop = target = 0.0
        direction = 0

        # Start after warmup period
        start_idx = max(self.MASS_LEN, self.STRUCT_LEN, 50)

        for i in range(start_idx, len(df)):

            # Current Slice
            price = close.iloc[i]
            c_low = low.iloc[i]
            c_high = high.iloc[i]
            c_atr = atr.iloc[i]

            # Physics Values
            ke_now = signed_ke.iloc[i]
            decay = ke_decay.iloc[i]

            # Booleans
            is_fake_now = is_fake.iloc[i]
            struct_up_now = structure_up.iloc[i]
            struct_down_now = structure_down.iloc[i]
            res_now = resonance.iloc[i]

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
                    # Optional: Exit on Exhaustion Regime could go here

                else:  # SHORT
                    if c_high >= stop:
                        exit_res = "STOP"
                        exit_price = stop
                    elif c_low <= target:
                        exit_res = "WIN"
                        exit_price = target

                if exit_res:
                    pnl = (exit_price - entry) / entry * 100
                    if direction == -1:
                        pnl *= -1
                    pnl -= self.FEE

                    self.trades.append({
                        "result": exit_res,
                        "pnl": round(pnl, 2),
                        "entry": entry,
                        "exit": exit_price,
                        "index": i
                    })
                    position = None
                    continue

            # --------------------------------------------
            # ENTRY LOGIC (Regime Based)
            # --------------------------------------------
            if not position:

                # 1. Determine Regime (Auto-Switcher)
                regime = "IDLE"
                if abs(ke_now) < 0.5 and abs(decay) < 0.1:
                    regime = "COMPRESSION"
                elif abs(ke_now) > 1.5 and decay > 0:
                    regime = "EXPANSION"
                elif abs(ke_now) > 0.8 and decay >= -0.2:
                    regime = "TREND"
                elif decay < -0.5:
                    regime = "EXHAUSTION"

                bias = "HOLD"

                # 2. Decision Matrix
                if regime == "TREND":
                    if ke_now > 0 and (struct_up_now or res_now) and not is_fake_now:
                        bias = "LONG"
                    elif ke_now < 0 and (struct_down_now or res_now) and not is_fake_now:
                        bias = "SHORT"

                elif regime == "EXPANSION":
                    # Breakouts need higher energy threshold
                    if ke_now > 1.2 and not is_fake_now:
                        bias = "LONG"
                    elif ke_now < -1.2 and not is_fake_now:
                        bias = "SHORT"

                # 3. Execution
                if bias != "HOLD":
                    direction = 1 if bias == "LONG" else -1

                    # v21 Sizing Logic
                    stop_mult = 2.5 if regime == "EXPANSION" else 1.5

                    # Using "Target 2" from v21 as the backtest target
                    if regime == "EXPANSION":
                        target_mult = 3.0  # Quick scalp in volatility
                    else:
                        target_mult = 3.5  # Ride the trend

                    entry = price
                    stop = price - (direction * c_atr * stop_mult)
                    target = price + (direction * c_atr * target_mult)

                    position = bias

        return self._stats()

    # ------------------------------------------------
    # STATS GENERATION
    # ------------------------------------------------
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
            "recent_trades": self.trades[-20:]
        }

    def _empty_result(self):
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "recent_trades": []
        }