import numpy as np
import pandas as pd
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    BACKTEST ENGINE v20 – TRUE PHYSICS ALIGNMENT

    ✔ Signed Kinetic Energy
    ✔ Energy Decay Detection
    ✔ Structural Confirmation (HH / LL)
    ✔ Regime-Aware Entries
    ✔ Adaptive Risk
    ✔ Silence is VALID
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.trades = []

    # ------------------------------------------------
    # CORE RUN
    # ------------------------------------------------
    # FIXED: Added 'mode' parameter to match views.py call
    def run(self, mode="INTRADAY"):
        if self.df.empty:
            return self._empty_result()

        # You can pass 'mode' to feature engineering if needed,
        # otherwise it just prevents the TypeError.
        df = generate_features(self.df)

        # -------------------------------
        # PHYSICS CONSTRUCTION
        # -------------------------------
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        atr = df["atr_14"]

        # SIGNED KINETIC ENERGY (v20 FIX)
        velocity = close.diff()
        mass = volume / (volume.rolling(20).mean() + 1e-9)
        ke = mass * velocity  # SIGNED

        # ENERGY DECAY
        ke_slope = ke.diff(3)

        # STRUCTURE
        higher_high = high > high.shift(1)
        lower_low = low < low.shift(1)

        # TREND BASELINE
        baseline = close.ewm(span=50).mean()

        # REGIME
        regime = np.where(abs(ke) > ke.rolling(50).std(), "SURGE", "FLOW")

        position = None
        entry = stop = target = 0.0
        direction = 0

        FEE = 0.06  # Binance taker %

        # ------------------------------------------------
        # LOOP
        # ------------------------------------------------
        for i in range(60, len(df)):

            price = close.iloc[i]
            c_ke = ke.iloc[i]
            c_ke_slope = ke_slope.iloc[i]
            c_atr = atr.iloc[i]
            c_regime = regime[i]

            # --------------------------------------------
            # EXIT LOGIC (DECAY + STRUCTURE BREAK)
            # --------------------------------------------
            if position:

                exit_reason = None
                exit_price = price

                # STOP / TARGET
                if direction == 1:
                    if low.iloc[i] <= stop:
                        exit_reason = "STOP"
                        exit_price = stop
                    elif high.iloc[i] >= target:
                        exit_reason = "TARGET"

                else:
                    if high.iloc[i] >= stop:
                        exit_reason = "STOP"
                        exit_price = stop
                    elif low.iloc[i] <= target:
                        exit_reason = "TARGET"

                # ENERGY DECAY EXIT (CRITICAL v20 RULE)
                if c_ke_slope < 0:
                    exit_reason = "ENERGY_DECAY"

                # REGIME BREAK EXIT
                if c_regime == "FLOW" and abs(c_ke) < 0.3:
                    exit_reason = "REGIME_FADE"

                if exit_reason:
                    pnl = (exit_price - entry) / entry * 100
                    if direction == -1:
                        pnl *= -1
                    pnl -= FEE

                    self.trades.append({
                        "result": exit_reason,
                        "pnl": round(pnl, 2),
                        "entry": entry,
                        "exit": exit_price,
                        "index": i
                    })

                    position = None
                    continue

            # --------------------------------------------
            # ENTRY LOGIC (STRICT BUT SMART)
            # --------------------------------------------
            if not position:

                bias = "HOLD"

                # LONG CONDITIONS
                if (
                    c_ke > 0.5 and
                    c_ke_slope >= 0 and
                    higher_high.iloc[i] and
                    price > baseline.iloc[i]
                ):
                    bias = "LONG"

                # SHORT CONDITIONS
                elif (
                    c_ke < -0.5 and
                    c_ke_slope <= 0 and
                    lower_low.iloc[i] and
                    price < baseline.iloc[i]
                ):
                    bias = "SHORT"

                if bias == "HOLD":
                    continue  # SILENCE IS VALID

                # ----------------------------------------
                # ADAPTIVE RISK (v20)
                # ----------------------------------------
                direction = 1 if bias == "LONG" else -1

                energy_strength = min(2.0, max(0.8, abs(c_ke)))
                stop_dist = c_atr * (1.2 * energy_strength)
                target_dist = stop_dist * 2.5

                entry = price
                stop = price - direction * stop_dist
                target = price + direction * target_dist

                position = bias

        return self._stats()

    # ------------------------------------------------
    # STATS
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