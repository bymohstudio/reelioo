# core/quant/backtest_engine.py

from dataclasses import dataclass, field
from typing import List
import pandas as pd
import numpy as np


@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    outcome: str  # WIN / LOSS


@dataclass
class BacktestSummary:
    symbol: str
    total_trades: int
    win_rate: float
    net_profit_percent: float
    profit_factor: float
    max_drawdown_percent: float
    avg_trade_percent: float
    sharpe_ratio: float  # NEW: Institutional Risk Metric
    max_win_streak: int  # NEW: Psychological Metric
    max_loss_streak: int  # NEW: Risk Metric
    recent_trades: List[TradeRecord]


class BacktestEngine:
    """
    Institutional Event-Driven Backtester.
    Simulates the Quant Engine candle-by-candle over history.

    UPGRADES:
    - Commission & Slippage Simulation (Realism)
    - Sharpe Ratio Calculation
    - Streak Analysis
    """

    def __init__(self, engine_class, df: pd.DataFrame, symbol: str, trade_style: str = "SWING"):
        self.engine_class = engine_class
        self.df = df
        self.symbol = symbol
        self.trade_style = trade_style
        self.trades = []
        self.equity_curve = [100.0]  # Start with 100% capital

        # Institutional Cost Model (Indian Markets)
        # 0.03% Brokerage/STT + 0.02% Slippage = 0.05% per trade
        self.cost_per_trade = 0.05

    def run(self, start_idx: int = 50) -> BacktestSummary:
        if len(self.df) < start_idx + 10:
            return self._empty()

        in_position = False
        entry_price = 0.0
        direction = ""
        entry_date = ""

        # Event-Driven Loop
        for i in range(start_idx, len(self.df)):
            current_slice = self.df.iloc[:i + 1]
            current_bar = self.df.iloc[i]

            # 1. Exit Logic
            if in_position:
                exit_signal = False
                exit_price = 0.0

                if direction == "BUY":
                    if current_bar['low'] < self.stop_loss:  # Stop Hit
                        exit_price = self.stop_loss
                        exit_signal = True
                    elif current_bar['high'] > self.target:  # Target Hit
                        exit_price = self.target
                        exit_signal = True
                elif direction == "SELL":
                    if current_bar['high'] > self.stop_loss:
                        exit_price = self.stop_loss
                        exit_signal = True
                    elif current_bar['low'] < self.target:
                        exit_price = self.target
                        exit_signal = True

                if exit_signal:
                    # Gross PnL
                    if direction == "BUY":
                        gross_pnl = (exit_price - entry_price) / entry_price
                    else:
                        gross_pnl = (entry_price - exit_price) / entry_price

                    # Net PnL (After Costs)
                    # We deduct cost twice (Entry + Exit) = 0.1% total round trip friction
                    net_pnl = gross_pnl - (self.cost_per_trade * 2 / 100)

                    outcome = "WIN" if net_pnl > 0 else "LOSS"

                    self.trades.append(TradeRecord(
                        str(entry_date), str(current_bar.name), direction,
                        entry_price, exit_price, round(net_pnl * 100, 2), outcome
                    ))

                    # Update Equity
                    self.equity_curve.append(self.equity_curve[-1] * (1 + net_pnl))
                    in_position = False
                    continue

            # 2. Entry Logic
            if not in_position:
                # Use simplified Trend Logic for speed
                # (Running full XGBoost 5000 times would be too slow for HTTP request)
                from core.quant.base_engine import compute_trend_score, build_entry_target_stop
                trend = compute_trend_score(current_slice)

                sig = "NEUTRAL"
                # Strategy: Trend Following
                if trend > 40:
                    sig = "BUY"
                elif trend < -40:
                    sig = "SELL"

                if sig != "NEUTRAL":
                    in_position = True
                    direction = sig
                    entry_price = float(current_bar['close'])
                    entry_date = current_bar.name

                    _, t, s, _ = build_entry_target_stop(current_slice, sig, self.trade_style)
                    self.target = t
                    self.stop_loss = s

        return self._calculate_stats()

    def _calculate_stats(self):
        if not self.trades: return self._empty()

        wins = [t for t in self.trades if t.outcome == "WIN"]
        losses = [t for t in self.trades if t.outcome == "LOSS"]

        win_rate = (len(wins) / len(self.trades)) * 100
        net_pnl = self.equity_curve[-1] - 100.0

        gross_profit = sum(t.pnl_percent for t in wins)
        gross_loss = abs(sum(t.pnl_percent for t in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9

        # Max Drawdown
        peaks = np.maximum.accumulate(self.equity_curve)
        drawdowns = (self.equity_curve - peaks) / peaks
        max_dd = drawdowns.min() * 100

        # Sharpe Ratio (Simplified Annualized)
        returns = pd.Series([t.pnl_percent for t in self.trades])
        sharpe = 0.0
        if returns.std() > 0:
            # Assuming ~252 trading days, scaled by trade frequency
            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(self.trades))

        # Streak Analysis
        outcomes = [1 if t.outcome == "WIN" else 0 for t in self.trades]

        def get_streak(val):
            max_s = curr = 0
            for x in outcomes:
                if x == val:
                    curr += 1
                else:
                    max_s = max(max_s, curr); curr = 0
            return max(max_s, curr)

        return BacktestSummary(
            symbol=self.symbol,
            total_trades=len(self.trades),
            win_rate=round(win_rate, 1),
            net_profit_percent=round(net_pnl, 1),
            profit_factor=profit_factor,
            max_drawdown_percent=round(max_dd, 1),
            avg_trade_percent=round(net_pnl / len(self.trades), 2),
            sharpe_ratio=round(sharpe, 2),
            max_win_streak=get_streak(1),
            max_loss_streak=get_streak(0),
            recent_trades=self.trades[-5:]
        )

    def _empty(self):
        return BacktestSummary(self.symbol, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, [])