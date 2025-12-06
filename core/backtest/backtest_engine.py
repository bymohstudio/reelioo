from dataclasses import dataclass, field
from typing import List, Optional, Type, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Import your engines to type check or use
from core.quant.base_engine import SignalResult

@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    symbol: str
    direction: str  # "LONG" / "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    pnl_percent: float
    outcome: str    # "WIN", "LOSS", "TIMEOUT"
    score_at_entry: float

@dataclass
class BacktestSummary:
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit_percent: float
    avg_trade_percent: float
    max_drawdown_percent: float
    profit_factor: float
    recent_trades: List[TradeRecord]

class BacktestEngine:
    """
    Institutional Event-Driven Backtester.
    Iterates through historical data, generating signals using the
    actual Quant Engine logic at each step to prevent look-ahead bias.
    """

    def __init__(self, engine_class, df: pd.DataFrame, symbol: str, trade_style: str = "SWING"):
        self.engine = engine_class
        self.df = df
        self.symbol = symbol
        self.trade_style = trade_style
        self.trades: List[TradeRecord] = []
        self.equity_curve = [100.0] # Normalize start at 100%

    def run(self, start_idx: int = 50) -> BacktestSummary:
        """
        Run the simulation.
        start_idx: skip initial bars to allow indicators (MA200 etc) to warm up.
        """
        if len(self.df) < start_idx + 10:
            return self._empty_result()

        active_trade = None
        
        # Loop through history
        # We step forward one bar at a time
        for i in range(start_idx, len(self.df) - 1):
            # Window of data available at time 'i'
            # We copy to prevent SettingWithCopy warnings and simulate real-time state
            current_window = self.df.iloc[:i+1].copy()
            next_candle = self.df.iloc[i+1]
            
            current_date = current_window.index[-1]
            
            # --- 1. MANAGE ACTIVE TRADE ---
            if active_trade:
                outcome = self._check_exit(active_trade, next_candle)
                if outcome:
                    active_trade['exit_date'] = next_candle.name.strftime("%Y-%m-%d %H:%M")
                    active_trade['exit_price'] = outcome['price']
                    active_trade['outcome'] = outcome['type']
                    
                    # Calculate PnL
                    if active_trade['direction'] == "LONG":
                        pnl = (outcome['price'] - active_trade['entry_price']) / active_trade['entry_price']
                    else:
                        pnl = (active_trade['entry_price'] - outcome['price']) / active_trade['entry_price']
                        
                    active_trade['pnl_percent'] = round(pnl * 100, 2)
                    
                    # Log Trade
                    record = TradeRecord(**active_trade)
                    self.trades.append(record)
                    self.equity_curve.append(self.equity_curve[-1] * (1 + pnl))
                    
                    active_trade = None # Trade closed
                else:
                    # Trade continues
                    continue 

            # --- 2. SCAN FOR NEW TRADE (If none active) ---
            # Run the Quant Engine on the current window
            try:
                # We suppress console logs from engine to keep backtest clean
                signal: SignalResult = self.engine.run(current_window, self.symbol, self.trade_style)
            except Exception:
                continue

            # Entry Logic: High Confidence Only
            # We simulate "Limit Order" logic:
            # For backtesting simplicity, we assume Entry at Open of NEXT candle
            # to be conservative (slippage simulation).
            
            if signal.direction == "UP" and signal.score >= 70:
                active_trade = {
                    "entry_date": next_candle.name.strftime("%Y-%m-%d %H:%M"),
                    "symbol": self.symbol,
                    "direction": "LONG",
                    "entry_price": next_candle["Open"],
                    "stop_loss": signal.stop,
                    "target": signal.target,
                    "score_at_entry": signal.score,
                    # Placeholders
                    "exit_date": "", "exit_price": 0.0, "pnl_percent": 0.0, "outcome": ""
                }
            
            elif signal.direction == "DOWN" and signal.score <= 30:
                # Assuming Shorting is allowed (Futures/Crypto/Forex)
                # For Equity Spot, we might restrict this, but let's allow for generic testing
                active_trade = {
                    "entry_date": next_candle.name.strftime("%Y-%m-%d %H:%M"),
                    "symbol": self.symbol,
                    "direction": "SHORT",
                    "entry_price": next_candle["Open"],
                    "stop_loss": signal.stop,
                    "target": signal.target,
                    "score_at_entry": signal.score,
                    # Placeholders
                    "exit_date": "", "exit_price": 0.0, "pnl_percent": 0.0, "outcome": ""
                }

        return self._compile_results()

    def _check_exit(self, trade, candle):
        """
        Checks if the next candle hits SL or TP.
        Order of precedence: Assumes SL hit first in a wide-range bar (Conservative).
        """
        low = candle["Low"]
        high = candle["High"]
        
        if trade["direction"] == "LONG":
            # Check Stop Loss
            if low <= trade["stop_loss"]:
                return {"type": "LOSS", "price": trade["stop_loss"]}
            # Check Target
            if high >= trade["target"]:
                return {"type": "WIN", "price": trade["target"]}
                
        elif trade["direction"] == "SHORT":
            # Check Stop Loss
            if high >= trade["stop_loss"]:
                return {"type": "LOSS", "price": trade["stop_loss"]}
            # Check Target
            if low <= trade["target"]:
                return {"type": "WIN", "price": trade["target"]}
                
        return None # No exit

    def _compile_results(self) -> BacktestSummary:
        if not self.trades:
            return self._empty_result()
            
        wins = len([t for t in self.trades if t.outcome == "WIN"])
        losses = len([t for t in self.trades if t.outcome == "LOSS"])
        total = len(self.trades)
        
        win_rate = (wins / total * 100) if total > 0 else 0.0
        
        net_pnl = sum(t.pnl_percent for t in self.trades)
        
        # Profit Factor
        gross_profit = sum(t.pnl_percent for t in self.trades if t.pnl_percent > 0)
        gross_loss = abs(sum(t.pnl_percent for t in self.trades if t.pnl_percent < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
        
        # Max Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100 # In percent
        
        return BacktestSummary(
            symbol=self.symbol,
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 1),
            net_profit_percent=round(net_pnl, 2),
            avg_trade_percent=round(net_pnl / total, 2),
            max_drawdown_percent=round(max_dd, 2),
            profit_factor=round(pf, 2),
            recent_trades=self.trades[-5:] # Last 5
        )

    def _empty_result(self):
        return BacktestSummary(
            self.symbol, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, []
        )