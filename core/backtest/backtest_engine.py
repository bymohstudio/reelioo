# core/quant/backtest_engine.py

import numpy as np
import pandas as pd
import logging
from core.quant.feature_engineering import generate_features

log = logging.getLogger(__name__)


def cap(x, limit=3.0):
    return max(-limit, min(limit, x))


class CryptoBacktestEngine:
    """
    PHYSICS BACKTEST ENGINE (v16.0 - Aligned with Fractal Geometry)
    - Updated to match CryptoQuantEngine v16.0 logic exactly.
    - Uses Vectorized Pandas for speed.
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []
        self.SIGMOID_K = 0.5  # Updated to match v16
        self.ATTACK_THRESH = 80
        self.ENGAGE_THRESH = 65

        # Gates
        self.MIN_VOLUME_RATIO = 0.6
        self.MAX_ATR_PCT = 0.05

    def _sigmoid(self, x):
        return 100 / (1 + np.exp(-self.SIGMOID_K * x))

    def run(self, trade_style="INTRADAY"):
        try:
            if self.df is None or self.df.empty: return self._empty_result()

            # 1. Feature Engineering
            try:
                df = generate_features(self.df.copy())
            except:
                return self._empty_result()

            # --------------------------------------------------------------
            # REPLICATE v16.0 LOGIC (VECTORIZED)
            # --------------------------------------------------------------

            # A. PHYSICS
            # Mass & Velocity
            mass = df.get('quote_volume', df['volume'])
            velocity = df['close'].diff()
            acceleration = velocity.diff()
            jerk = acceleration.diff()

            # B. FRACTAL EFFICIENCY (ER)
            # ER = Change / Sum of absolute changes (Period 10)
            period = 10
            change = df['close'].diff(period).abs()
            volatility = df['close'].diff().abs().rolling(period).sum()
            efficiency_ratio = change / (volatility + 0.00001)

            # C. FORCE DIVERGENCE
            force = mass * velocity
            # Price > 5 bars ago AND Force < 5 bars ago
            price_trend_5 = df['close'] > df['close'].shift(5)
            force_trend_5 = force < force.shift(5)
            is_divergence = price_trend_5 & force_trend_5

            # D. VECTORS
            # Trend Vector (Fractal Adjusted)
            raw_trend = df['ema_diff'].clip(-0.03, 0.03) * 100 * 2.0
            # Vectorized conditional: if ER > 0.5 then 1.5 else 0.5
            fractal_quality = np.where(efficiency_ratio > 0.5, 1.5, 0.5)
            trend_alpha = raw_trend * fractal_quality

            # Whale Vector
            whale_z = df['whale_z'].clip(-3, 3)

            # Kinetic Snap
            # Replicating: if abs(kinetic) > 1.0: alpha += jerk*10 + kinetic*0.5
            kinetic_energy = df['kinetic_energy']
            jerk_impact = jerk.clip(-0.3, 0.3) * 10.0

            physics_alpha = np.zeros(len(df))
            mask_kinetic = kinetic_energy.abs() > 1.0
            physics_alpha[mask_kinetic] = jerk_impact[mask_kinetic] + (kinetic_energy[mask_kinetic] * 0.5)

            # Spring Compression
            # if comp < 0.6 and whale > 0.8: alpha += 4.0 * sign(jerk + trend)
            compression = df['volatility_compression']
            is_spring = (compression < 0.6) & (whale_z.abs() > 0.8)

            breakout_dir = np.sign(jerk_impact + trend_alpha)
            # Add spring boost where applicable
            physics_alpha = np.where(is_spring, physics_alpha + (4.0 * breakout_dir), physics_alpha)

            # E. SCORING
            total_alpha = trend_alpha + whale_z + physics_alpha
            # Apply Sigmoid manually to array
            raw_score = 100 / (1 + np.exp(-self.SIGMOID_K * total_alpha))

            # F. GATES & PENALTIES
            # Fakeout Penalty
            # if vol_slope > 0.25 and kinetic < 0.8: penalty 30
            vol_slope = df['volatility_slope']
            penalty = np.where((vol_slope > 0.25) & (kinetic_energy.abs() < 0.8), 30, 0)

            # Force Divergence Penalty
            penalty = np.where(is_divergence & (~is_spring), penalty + 25, penalty)

            final_score = raw_score - penalty

            # G. HARD GATES (Zero out score if gate closed)
            # Liquidity Gate
            avg_vol = df['volume'].rolling(20).mean()
            gate_liquid = df['volume'] >= (avg_vol * self.MIN_VOLUME_RATIO)

            # Fractal Noise Gate
            gate_fractal = (efficiency_ratio >= 0.25) | is_spring

            # Volatility Gate
            atr_pct = df['atr_pct']
            gate_vol = (atr_pct <= self.MAX_ATR_PCT) | is_spring

            # Combine Gates
            open_gates = gate_liquid & gate_fractal & gate_vol

            # Final Signal Series (0 to 100, or 50 if closed)
            final_score = np.where(open_gates, final_score, 50)

            # --------------------------------------------------------------
            # EXECUTION LOOP
            # --------------------------------------------------------------
            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0
            TRADING_FEE_PCT = 0.1

            start_idx = 50  # Allow indicators to warm up

            for i in range(start_idx, len(df)):
                curr_score = final_score[i]
                price = df['close'].iloc[i]
                low = df['low'].iloc[i]
                high = df['high'].iloc[i]
                atr = df['atr_14'].iloc[i]
                curr_er = efficiency_ratio.iloc[i]

                # --- EXIT LOGIC ---
                if position:
                    res = None
                    exit_price = price
                    if position == 'LONG':
                        if low <= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif high >= take_profit:
                            res, exit_price = "WIN", take_profit
                    elif position == 'SHORT':
                        if high >= stop_loss:
                            res, exit_price = "LOSS", stop_loss
                        elif low <= take_profit:
                            res, exit_price = "WIN", take_profit

                    if res:
                        pnl = (exit_price - entry_price) / entry_price * 100
                        if position == 'SHORT': pnl = -pnl
                        self.trades.append({
                            "result": res,
                            "pnl": round(pnl - TRADING_FEE_PCT, 2),
                            "entry": entry_price,
                            "date": str(df.index[i])
                        })
                        position = None
                        continue

                # --- ENTRY LOGIC (Mapped from v16 Lanes) ---
                if not position:
                    bias = "HOLD"

                    if curr_score >= self.ATTACK_THRESH:
                        bias = "LONG"
                    elif curr_score <= (100 - self.ATTACK_THRESH):
                        bias = "SHORT"
                    elif curr_score >= self.ENGAGE_THRESH:
                        bias = "LONG"
                    elif curr_score <= (100 - self.ENGAGE_THRESH):
                        bias = "SHORT"

                    if bias != "HOLD":
                        direction = 1 if bias == 'LONG' else -1

                        # Dynamic Extension (Fractal)
                        fractal_mult = 1.5 if curr_er > 0.6 else 1.0

                        entry_price = price
                        # Stop is 2.0 ATR in v16
                        stop_loss = price - direction * (atr * 2.0)
                        # Target 2 is 4.0 * fractal_mult
                        take_profit = price + direction * (atr * 4.0 * fractal_mult)

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