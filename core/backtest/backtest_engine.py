# core/quant/backtest_engine.py

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    outcome: str


@dataclass
class BacktestSummary:
    symbol: str
    total_trades: int
    win_rate: float
    net_profit_percent: float
    profit_factor: float
    max_drawdown_percent: float
    avg_trade_percent: float
    sharpe_ratio: float
    max_win_streak: int
    max_loss_streak: int
    recent_trades: List[TradeRecord]


class BacktestEngine:
    """
    Institutional Event-Driven Backtester.
    """

    def __init__(self, engine_class, df: pd.DataFrame, symbol: str, trade_style: str = "SWING"):
        self.engine_class = engine_class
        self.df = df
        self.symbol = symbol
        self.trade_style = trade_style
        self.trades: List[TradeRecord] = []
        self.equity_curve = []

    def run(self, start_idx: int = 50) -> BacktestSummary:
        if self.df is None or len(self.df) < start_idx + 10:
            return BacktestSummary(self.symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        position = None
        entry_price = 0.0
        stop_loss = 0.0
        target_price = 0.0
        COMMISSION = 0.0005  # 0.05% per trade

        # Iterate through history
        # We start from 'start_idx' to allow indicators to warm up
        for i in range(start_idx, len(self.df) - 1):
            curr_row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]

            # --- 1. Position Management (Exit Logic) ---
            if position:
                exit_price = 0.0
                is_closed = False

                # Check High/Low of NEXT candle to see if Stop/Target hit
                if position == "LONG":
                    if next_row['low'] <= stop_loss:
                        exit_price = stop_loss
                        is_closed = True
                    elif next_row['high'] >= target_price:
                        exit_price = target_price
                        is_closed = True
                    # Time-based exit for intraday (optional, simplified here)

                    if is_closed:
                        pnl = ((exit_price - entry_price) / entry_price) - (2 * COMMISSION)
                        self._record_trade(curr_row.name, next_row.name, "LONG", entry_price, exit_price, pnl)
                        position = None
                        continue

                elif position == "SHORT":
                    if next_row['high'] >= stop_loss:
                        exit_price = stop_loss
                        is_closed = True
                    elif next_row['low'] <= target_price:
                        exit_price = target_price
                        is_closed = True

                    if is_closed:
                        pnl = ((entry_price - exit_price) / entry_price) - (2 * COMMISSION)
                        self._record_trade(curr_row.name, next_row.name, "SHORT", entry_price, exit_price, pnl)
                        position = None
                        continue

            # --- 2. Entry Logic ---
            if not position:
                # IMPORTANT: Pass the SLICE of data up to current time 'i'
                # The engine needs history to calc indicators (MA, RSI)
                # But it MUST NOT see the future (i+1)

                # Performance optimization: Don't re-run engine on every single candle if possible
                # But for accuracy, we must.
                # We limit the slice to last 200 candles to speed up
                history_slice = self.df.iloc[max(0, i - 200):i + 1].copy()

                try:
                    # Run the strategy engine on this historical slice
                    res = self.engine_class.run(history_slice, self.symbol, self.trade_style)

                    # Entry Threshold (Slightly relaxed for backtest to show data)
                    if res.score >= 55:
                        if res.direction in ["BUY", "BULLISH"]:
                            position = "LONG"
                            entry_price = next_row['open']  # Enter on next open
                            stop_loss = res.stop
                            target_price = res.target
                        elif res.direction in ["SELL", "BEARISH"]:
                            position = "SHORT"
                            entry_price = next_row['open']
                            stop_loss = res.stop
                            target_price = res.target

                except Exception as e:
                    # Log only once to avoid spam
                    if i == start_idx: log.error(f"Backtest Engine Step Failed: {e}")
                    continue

        return self._generate_summary()

    def _record_trade(self, entry_dt, exit_dt, direction, entry, exit, pnl):
        self.trades.append(TradeRecord(
            entry_date=str(entry_dt),
            exit_date=str(exit_dt),
            direction=direction,
            entry_price=entry,
            exit_price=exit,
            pnl_percent=pnl * 100,
            outcome="WIN" if pnl > 0 else "LOSS"
        ))
        self.equity_curve.append(pnl * 100)

    def _generate_summary(self):
        total = len(self.trades)
        if total == 0:
            return BacktestSummary(self.symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        wins = [t for t in self.trades if t.outcome == "WIN"]
        losses = [t for t in self.trades if t.outcome == "LOSS"]

        win_rate = (len(wins) / total) * 100
        net_pnl = sum([t.pnl_percent for t in self.trades])

        avg_win = np.mean([t.pnl_percent for t in wins]) if wins else 0
        avg_loss = np.abs(np.mean([t.pnl_percent for t in losses])) if losses else 0
        pf = (avg_win * len(wins)) / (avg_loss * len(losses)) if (avg_loss * len(losses)) > 0 else 0

        # Max Drawdown
        eq = np.array(self.equity_curve).cumsum()
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq).max() if len(peak) > 0 else 0

        # Sharpe
        returns = pd.Series([t.pnl_percent for t in self.trades])
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        return BacktestSummary(
            symbol=self.symbol,
            total_trades=total,
            win_rate=round(win_rate, 1),
            net_profit_percent=round(net_pnl, 1),
            profit_factor=round(pf, 2),
            max_drawdown_percent=round(dd, 1),
            avg_trade_percent=round(net_pnl / total, 2),
            sharpe_ratio=round(sharpe, 2),
            max_win_streak=0,  # Simplified
            max_loss_streak=0,
            recent_trades=self.trades[-5:]
        )