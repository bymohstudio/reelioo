import numpy as np
import pandas as pd
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """
    PHYSICS BACKTEST ENGINE (v18.0 - KINETIC POTENTIAL)
    - Perfectly aligned with CryptoQuantEngine v18.0.
    - Simulates the "Projectile Motion" targeting logic.
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []

        # Physics Constants (Match Engine)
        self.MASS_WINDOW = 20
        self.VELOCITY_WINDOW = 3
        self.GRAVITY = 9.8

    def run(self, trade_style="INTRADAY"):
        try:
            if self.df is None or self.df.empty: return self._empty_result()

            # 1. Feature Engineering
            try:
                # We need ATR and basic features
                df = generate_features(self.df.copy())
            except:
                return self._empty_result()

            # --------------------------------------------------------------
            # REPLICATE v18.0 PHYSICS LOGIC (VECTORIZED)
            # --------------------------------------------------------------

            # A. CONSTRUCT PHYSICS WORLD
            close = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']

            # Mass (m)
            vol_avg = volume.rolling(self.MASS_WINDOW).mean()
            mass = volume / (vol_avg + 0.0001)

            # Velocity (v)
            # Normalized by ATR to make it asset-agnostic
            atr = df['atr_14']
            velocity = close.diff(self.VELOCITY_WINDOW) / (atr + 0.0001)

            # Acceleration (a)
            acceleration = velocity.diff()

            # Force (F = ma)
            force = mass * acceleration

            # Kinetic Energy (KE = 0.5 * m * v^2)
            ke = 0.5 * mass * (velocity ** 2)

            # Potential Energy (PE)
            # Inverse normalized volatility
            norm_vol = atr / close
            pe_raw = (1.0 / (norm_vol + 0.001)).rolling(10).mean()
            pe_min = pe_raw.rolling(50).min()
            pe_max = pe_raw.rolling(50).max()
            pe_score = (pe_raw - pe_min) / (pe_max - pe_min + 0.001) * 100

            # Baseline (Gravity)
            baseline = close.ewm(span=50).mean()

            # B. IDENTIFY STATES (Boolean Masks)
            is_compressed = (pe_score > 80) & (ke < 1.0)
            is_exploding = (ke > 2.0) & (mass > 1.2)
            is_trending = (ke > 1.0) & (velocity.abs() > 0.5)

            # C. SCORING VECTORS
            score = pd.Series(50, index=df.index, dtype='float64')

            # Vector 1: Force Alignment
            # Force > 0 and Velo > 0 (Pushing Up)
            score += np.where((force > 0) & (velocity > 0), 15, 0)
            # Force < 0 and Velo < 0 (Pushing Down)
            score += np.where((force < 0) & (velocity < 0), 15, 0)

            # Vector 2: Mass Confirmation
            score += np.where(mass > 1.5, 15, 0)
            score -= np.where(mass < 0.5, 10, 0)

            # Vector 3: Kinetic State
            score += np.where(is_exploding, 25, 0)
            score += np.where(is_trending, 10, 0)

            # Vector 4: Potential (Compression Bias)
            # If compressed, bias slightly towards current velocity drift
            score += np.where(is_compressed & (velocity > 0), 5, 0)
            score += np.where(is_compressed & (velocity < 0), 5, 0)

            # D. DECISION LOGIC
            # Direction
            # Long: Velocity > 0 and Price > Baseline
            # Short: Velocity < 0 and Price < Baseline

            # Normalize Score
            score = score.clip(1, 99)

            # --------------------------------------------------------------
            # SIMULATION LOOP (Trade Management)
            # --------------------------------------------------------------
            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            TRADING_FEE_PCT = 0.06  # Binance Taker Fee approx

            start_idx = 50

            for i in range(start_idx, len(df)):
                c_score = score.iloc[i]
                c_price = close.iloc[i]
                c_low = low.iloc[i]
                c_high = high.iloc[i]
                c_atr = atr.iloc[i]
                c_ke = ke.iloc[i]
                c_base = baseline.iloc[i]
                c_velo = velocity.iloc[i]

                # --- EXIT LOGIC ---
                if position:
                    res = None
                    exit_price = c_price

                    if position == 'LONG':
                        if c_low <= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif c_high >= take_profit:
                            res, exit_price = "WIN", take_profit

                    elif position == 'SHORT':
                        if c_high >= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif c_low <= take_profit:
                            res, exit_price = "WIN", take_profit

                    if res:
                        # Calculate PnL
                        raw_pnl = (exit_price - entry_price) / entry_price * 100
                        if position == 'SHORT': raw_pnl = -raw_pnl

                        net_pnl = raw_pnl - TRADING_FEE_PCT

                        self.trades.append({
                            "result": res,
                            "pnl": round(net_pnl, 2),
                            "entry": entry_price,
                            "date": str(df.index[i])
                        })
                        position = None
                        continue

                # --- ENTRY LOGIC (Physics Based) ---
                if not position:
                    bias = "HOLD"

                    # Lane Logic from Engine
                    if c_score >= 80:  # POWER Zone
                        # Direction Check
                        if c_velo > 0 and c_price > c_base:
                            bias = "LONG"
                        elif c_velo < 0 and c_price < c_base:
                            bias = "SHORT"

                    elif c_score >= 60:  # Active Zone
                        if c_velo > 0 and c_price > c_base:
                            bias = "LONG"
                        elif c_velo < 0 and c_price < c_base:
                            bias = "SHORT"

                    # Trade Execution
                    if bias != "HOLD":
                        direction = 1 if bias == 'LONG' else -1

                        # Projectile Physics Targeting
                        # Stop: 1.5 ATR (Where momentum breaks)
                        stop_dist = c_atr * 1.5

                        # Throw Power: Based on Kinetic Energy (Cap at 3x)
                        throw_power = max(1.0, min(3.0, c_ke))

                        # Target: 3.0 ATR * Power
                        # We use Target 2 from the engine as the primary exit for backtest
                        target_dist = c_atr * 3.0 * throw_power

                        entry_price = c_price
                        stop_loss = c_price - (direction * stop_dist)
                        take_profit = c_price + (direction * target_dist)

                        position = bias

            return self._generate_stats()

        except Exception as e:
            log.error(f"Backtest Error: {e}")
            return self._empty_result()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0: return self._empty_result()
        wins = [t for t in self.trades if t['pnl'] > 0]
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 10.0
        return {
            "win_rate": round(len(wins) / total * 100, 1),
            "profit_factor": round(pf, 2),
            "total_trades": total,
            "trades_log": self.trades[-20:]
        }

    def _empty_result(self):
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []}