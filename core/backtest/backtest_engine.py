import pandas as pd
import numpy as np


class CryptoBacktestEngine:
    """
    Titanium Backtester (Layered Intelligence).
    Validates both 'Strong Trend' (Expansion) and 'Micro Scalp' (Flow) logic.
    Mirrors the risk management of the live crypto_engine.py.
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.trades = []

        # Calculate Indicators immediately
        self._prepare_indicators()

    def _prepare_indicators(self):
        """
        Calculates the exact Technical Indicators used by the live engine.
        """
        # 1. EMAs for Trend Flow
        self.df['ema_9'] = self.df['close'].ewm(span=9).mean()
        self.df['ema_21'] = self.df['close'].ewm(span=21).mean()
        self.df['ema_50'] = self.df['close'].ewm(span=50).mean()

        # 2. ATR for Dynamic Stops
        self.df['tr'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                abs(self.df['high'] - self.df['close'].shift(1)),
                abs(self.df['low'] - self.df['close'].shift(1))
            )
        )
        self.df['atr'] = self.df['tr'].rolling(14).mean()

        # 3. Volatility Z-Score (Approximating AI "Expansion" Feature)
        vol_mean = self.df['volume'].rolling(20).mean()
        vol_std = self.df['volume'].rolling(20).std()
        self.df['vol_z'] = (self.df['volume'] - vol_mean) / (vol_std + 1e-9)

        # 4. Trend Strength
        self.df['trend_strength'] = (self.df['ema_9'] - self.df['ema_21']) / self.df['close'] * 100

        # Fill NaNs
        self.df.fillna(0, inplace=True)

    def run(self):
        """
        Simulates the 'Layered Intelligence' strategy on historical data.
        """
        position = None
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        signal_type = "NONE"

        # Skip warm-up period for EMAs
        start_idx = 50

        for i in range(start_idx, len(self.df)):
            curr = self.df.iloc[i]
            # prev = self.df.iloc[i - 1] # Can use for crossovers if needed

            price = float(curr['close'])
            atr = float(curr['atr']) if curr['atr'] > 0 else price * 0.01

            # --- 1. EXIT LOGIC ---
            if position:
                exit_price = 0.0
                result = ""

                # LONG EXIT
                if position == 'LONG':
                    if curr['low'] <= stop_loss:
                        exit_price = stop_loss
                        result = "LOSS"
                    elif curr['high'] >= take_profit:
                        exit_price = take_profit
                        result = "WIN"

                # SHORT EXIT
                elif position == 'SHORT':
                    if curr['high'] >= stop_loss:
                        exit_price = stop_loss
                        result = "LOSS"
                    elif curr['low'] <= take_profit:
                        exit_price = take_profit
                        result = "WIN"

                # Process Trade Record
                if exit_price != 0.0:
                    if position == 'LONG':
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        # Short PnL: (Entry - Exit) / Entry
                        pnl = (entry_price - exit_price) / entry_price

                    self.trades.append({
                        "type": position,
                        "style": signal_type,  # 'STRONG' or 'SCALP'
                        "entry": round(entry_price, 4),
                        "exit": round(exit_price, 4),
                        "pnl": round(pnl * 100, 2),
                        "result": result,
                        "date": str(curr.name)
                    })

                    # Reset
                    position = None
                    signal_type = "NONE"
                    continue

            # --- 2. ENTRY LOGIC (Matching Live Engine) ---
            if position is None:

                # A. STRONG TREND (Expansion Mode)
                # Logic: High Momentum (Trend Strength) + Volume Expansion (Vol Z)
                # This proxies the "ML High Confidence" signal
                is_strong_trend_up = (curr['trend_strength'] > 0.5) and (curr['vol_z'] > 1.0)
                is_strong_trend_down = (curr['trend_strength'] < -0.5) and (curr['vol_z'] > 1.0)

                # B. MICRO SCALP (Flow Mode)
                # Logic: Price respecting EMA hierarchy (Layer 2)
                # This proxies the "Speculative" signal
                is_flow_up = (curr['close'] > curr['ema_21']) and (curr['ema_21'] > curr['ema_50'])
                is_flow_down = (curr['close'] < curr['ema_21']) and (curr['ema_21'] < curr['ema_50'])

                # --- DECISION TREE ---

                # 1. LONG SIGNALS
                if is_strong_trend_up:
                    position = 'LONG'
                    signal_type = "STRONG"
                    stop_mult, target_mult = 2.0, 3.0  # Wide levels for Strong Trend
                elif is_flow_up:
                    position = 'LONG'
                    signal_type = "SCALP"
                    stop_mult, target_mult = 1.0, 1.5  # Tight levels for Scalp

                # 2. SHORT SIGNALS
                elif is_strong_trend_down:
                    position = 'SHORT'
                    signal_type = "STRONG"
                    stop_mult, target_mult = 2.0, 3.0
                elif is_flow_down:
                    position = 'SHORT'
                    signal_type = "SCALP"
                    stop_mult, target_mult = 1.0, 1.5

                # Execute Entry
                if position:
                    entry_price = price
                    if position == 'LONG':
                        stop_loss = price - (atr * stop_mult)
                        take_profit = price + (atr * target_mult)
                    else:
                        stop_loss = price + (atr * stop_mult)
                        take_profit = price - (atr * target_mult)

        return self._generate_stats()

    def _generate_stats(self):
        total = len(self.trades)
        if total == 0:
            return {
                "win_rate": 0, "profit_factor": 0, "total_trades": 0, "trades_log": []
            }

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = round((len(wins) / total) * 100, 1)

        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))

        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9

        # Return stats + log (Latest trades first)
        return {
            "win_rate": win_rate,
            "profit_factor": pf,
            "total_trades": total,
            "trades_log": self.trades[::-1][:50]
        }