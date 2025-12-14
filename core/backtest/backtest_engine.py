import pandas as pd
import numpy as np


class CryptoBacktestEngine:
    """
    Robust Backtester with CORRECT PnL Logic for Shorts.
    """

    def __init__(self, df, symbol):
        self.df = df
        self.symbol = symbol
        self.balance = 1000  # Starting Balance
        self.trades = []

        # Strategy Parameters
        self.atr_period = 14
        self.risk_reward = 1.5

        # Calculate Indicators needed for logic
        self._prepare_indicators()

    def _prepare_indicators(self):
        # Simple Moving Averages for Trend
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()

        # ATR for Volatility-based stops
        high_low = self.df['high'] - self.df['low']
        self.df['atr'] = high_low.rolling(self.atr_period).mean()

    def run(self):
        """
        Executes the simulation bar-by-bar.
        """
        position = None  # None, 'LONG', 'SHORT'
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        # Iterate through data (Skip first 50 for indicators to warm up)
        for i in range(50, len(self.df)):
            curr = self.df.iloc[i]
            prev = self.df.iloc[i - 1]

            price = float(curr['close'])
            atr = float(curr['atr']) if not pd.isna(curr['atr']) else price * 0.01

            # --- 1. EXIT LOGIC ---
            if position:
                pnl = 0
                result = ""
                exit_price = 0

                # Check Long Exits
                if position == 'LONG':
                    if curr['low'] <= stop_loss:
                        exit_price = stop_loss
                        pnl = (exit_price - entry_price) / entry_price
                        result = "LOSS"
                    elif curr['high'] >= take_profit:
                        exit_price = take_profit
                        pnl = (exit_price - entry_price) / entry_price
                        result = "WIN"

                # Check Short Exits (CRITICAL FIX)
                elif position == 'SHORT':
                    if curr['high'] >= stop_loss:
                        exit_price = stop_loss
                        # Short PnL: You make money if Entry > Exit
                        pnl = (entry_price - exit_price) / entry_price
                        result = "LOSS"
                    elif curr['low'] <= take_profit:
                        exit_price = take_profit
                        # Short PnL: You make money if Entry > Exit
                        pnl = (entry_price - exit_price) / entry_price
                        result = "WIN"

                # Record Trade if Exited
                if exit_price != 0:
                    pnl_percent = round(pnl * 100, 2)

                    # Double Check Result based on PnL Value (Safety Net)
                    real_result = "WIN" if pnl_percent > 0 else "LOSS"

                    self.trades.append({
                        "type": position,
                        "entry": round(entry_price, 2),
                        "exit": round(exit_price, 2),
                        "pnl": pnl_percent,
                        "result": real_result,
                        "date": str(curr.name)
                    })
                    position = None
                    continue

            # --- 2. ENTRY LOGIC ---
            if position is None:
                # Long Condition: SMA20 > SMA50
                if curr['sma_20'] > curr['sma_50'] and prev['sma_20'] <= prev['sma_50']:
                    position = 'LONG'
                    entry_price = price
                    stop_loss = price - (atr * 1.5)
                    take_profit = price + (atr * 1.5 * self.risk_reward)

                # Short Condition: SMA20 < SMA50
                elif curr['sma_20'] < curr['sma_50'] and prev['sma_20'] >= prev['sma_50']:
                    position = 'SHORT'
                    entry_price = price
                    stop_loss = price + (atr * 1.5)
                    take_profit = price - (atr * 1.5 * self.risk_reward)

        # --- 3. COMPILE STATS ---
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

        return {
            "win_rate": win_rate,
            "profit_factor": pf,
            "total_trades": total,
            "trades_log": self.trades[-50:]  # Return last 50
        }