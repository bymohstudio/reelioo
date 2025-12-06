import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.quant.backtest_engine import BacktestEngine, BacktestSummary
from core.quant.base_engine import SignalResult, Direction

# ==========================================
# MOCK & FIXTURES
# ==========================================

@dataclass
class MockSignal:
    """Helper to force specific signals from the MockEngine."""
    score: float
    direction: Direction
    entry: float
    target: float
    stop: float

class MockQuantEngine:
    """
    A controllable engine that returns pre-defined signals 
    based on the current timestamp of the backtest loop.
    """
    # Dictionary mapping Timestamp -> MockSignal
    signals = {}

    @classmethod
    def set_signals(cls, signal_map):
        cls.signals = signal_map

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str = "SWING") -> SignalResult:
        current_time = df.index[-1]
        
        # Default No Signal
        default_res = SignalResult(
            symbol=symbol, market="MOCK", style=trade_style, model_version="vMock",
            score=50.0, direction="FLAT", label="WAIT",
            entry=0.0, target=0.0, stop=0.0,
            support=0.0, resistance=0.0, time_frame="1D",
            expected_bars_to_target=0, expected_time_to_target_hours=0,
            factors={}, meta={}
        )

        if current_time in cls.signals:
            sig = cls.signals[current_time]
            return SignalResult(
                symbol=symbol, market="MOCK", style=trade_style, model_version="vMock",
                score=sig.score, direction=sig.direction, label="TEST_SIG",
                entry=sig.entry, target=sig.target, stop=sig.stop,
                support=0.0, resistance=0.0, time_frame="1D",
                expected_bars_to_target=0, expected_time_to_target_hours=0,
                factors={}, meta={}
            )
        
        return default_res

@pytest.fixture
def sample_data():
    """Generates a predictable price series for 100 days."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    data = {
        "Open": [100.0] * 100,
        "High": [105.0] * 100,
        "Low": [95.0] * 100,
        "Close": [100.0] * 100,
        "Volume": [1000] * 100
    }
    df = pd.DataFrame(data, index=dates)
    return df

# ==========================================
# TESTS
# ==========================================

def test_initialization_empty_data():
    """Test handling of empty or None dataframe."""
    engine = BacktestEngine(MockQuantEngine, pd.DataFrame(), "TEST")
    res = engine.run()
    assert res.total_trades == 0
    assert res.win_rate == 0.0

def test_insufficient_history(sample_data):
    """Test if backtest aborts gracefully when start_idx > data length."""
    # Data has 100 rows, start at 150
    engine = BacktestEngine(MockQuantEngine, sample_data, "TEST")
    res = engine.run(start_idx=150)
    assert res.total_trades == 0

def test_long_trade_win(sample_data):
    """
    Scenario:
    1. Signal BUY on row 50.
    2. Row 51 Open executes entry at 100.
    3. Row 52 High hits Target (110).
    """
    df = sample_data.copy()
    
    # Setup Signal at index 50 (2024-01-03 02:00:00)
    signal_time = df.index[50]
    
    # Define a WIN scenario
    # Entry 100, Target 110, Stop 90
    MockQuantEngine.set_signals({
        signal_time: MockSignal(score=80.0, direction="UP", entry=100, target=110, stop=90)
    })
    
    # Manipulate price action AFTER signal to ensure Hit
    # Row 51 is entry candle (execution)
    # Row 52 is result candle -> Make High 115 to trigger Target 110
    entry_idx = 51
    result_idx = 52
    
    df.iloc[result_idx, df.columns.get_loc("High")] = 115.0
    
    engine = BacktestEngine(MockQuantEngine, df, "TEST")
    res = engine.run(start_idx=40)
    
    assert res.total_trades == 1
    assert res.wins == 1
    assert res.win_rate == 100.0
    assert res.recent_trades[0].outcome == "WIN"
    # Profit: (110 - 100) / 100 = 10%
    assert res.recent_trades[0].pnl_percent == 10.0

def test_long_trade_loss(sample_data):
    """
    Scenario:
    1. Signal BUY on row 50.
    2. Row 51 Entry.
    3. Row 52 Low hits Stop Loss (90).
    """
    df = sample_data.copy()
    signal_time = df.index[50]
    
    MockQuantEngine.set_signals({
        signal_time: MockSignal(score=80.0, direction="UP", entry=100, target=110, stop=90)
    })
    
    # Make price crash on result candle
    df.iloc[52, df.columns.get_loc("Low")] = 85.0 # Breaches 90 SL
    
    engine = BacktestEngine(MockQuantEngine, df, "TEST")
    res = engine.run(start_idx=40)
    
    assert res.total_trades == 1
    assert res.losses == 1
    assert res.recent_trades[0].outcome == "LOSS"
    # Loss: (90 - 100) / 100 = -10%
    assert res.recent_trades[0].pnl_percent == -10.0

def test_short_trade_win(sample_data):
    """
    Scenario:
    1. Signal SELL on row 50 (Score 20, Direction DOWN).
    2. Row 51 Entry 100.
    3. Row 52 Low hits Target (90).
    """
    df = sample_data.copy()
    signal_time = df.index[50]
    
    MockQuantEngine.set_signals({
        signal_time: MockSignal(score=20.0, direction="DOWN", entry=100, target=90, stop=110)
    })
    
    # Make price drop
    df.iloc[52, df.columns.get_loc("Low")] = 85.0 # Hits Target 90
    
    engine = BacktestEngine(MockQuantEngine, df, "TEST")
    res = engine.run(start_idx=40)
    
    assert res.total_trades == 1
    assert res.wins == 1
    # Profit Short: (Entry 100 - Exit 90) / 100 = 10%
    assert res.recent_trades[0].pnl_percent == 10.0

def test_summary_metrics_calculation():
    """
    Simulate a sequence of trades to check Win Rate, PF, DD.
    Trade 1: +10%
    Trade 2: -5%
    Trade 3: +10%
    """
    # Create manual df not needed, we can mock logic, but let's use the engine
    # We will just inject results manually into a dummy result for testing logic
    # or simulate 3 distinct signals in time.
    
    # Let's verify the math in _compile_results by unit testing the Logic class directly
    # constructing a BacktestEngine with pre-filled trades.
    
    engine = BacktestEngine(MockQuantEngine, pd.DataFrame(), "TEST")
    
    from core.quant.backtest_engine import TradeRecord
    
    engine.trades = [
        TradeRecord("2024-01-01", "2024-01-02", "TEST", "LONG", 100, 110, 90, 110, 10.0, "WIN", 80),
        TradeRecord("2024-01-03", "2024-01-04", "TEST", "LONG", 100, 95, 95, 110, -5.0, "LOSS", 80),
        TradeRecord("2024-01-05", "2024-01-06", "TEST", "SHORT", 100, 90, 110, 90, 10.0, "WIN", 20),
    ]
    
    # Mock equity curve simulation [100, 110, 104.5, 114.95]
    # Trade 1 (+10%): 100 -> 110
    # Trade 2 (-5%): 110 -> 104.5
    # Trade 3 (+10%): 104.5 -> 114.95
    engine.equity_curve = [100.0, 110.0, 104.5, 114.95]
    
    res = engine._compile_results()
    
    assert res.total_trades == 3
    assert res.wins == 2
    assert res.losses == 1
    assert res.win_rate == 66.7 # 2/3
    assert res.net_profit_percent == 15.0 # Sum of simple pnl% (10 - 5 + 10)
    
    # Profit Factor: Gross Win (20) / Gross Loss (5) = 4.0
    assert res.profit_factor == 4.0
    
    # Drawdown: Peak was 110, dropped to 104.5. DD = (104.5 - 110)/110 = -5%
    # Note: Max DD is positive number in representation usually or negative. Code says min().
    # engine calculation: (equity - peak) / peak. Min value is -0.05. * 100 = -5.0.
    assert res.max_drawdown_percent == -5.0

def test_conflict_signal_ignored(sample_data):
    """
    Test that if a trade is active, new signals are ignored.
    """
    df = sample_data.copy()
    
    # Signal 1 at index 50 (Valid)
    # Signal 2 at index 51 (Should be ignored because Trade 1 is open)
    
    sig1_time = df.index[50]
    sig2_time = df.index[51]
    
    MockQuantEngine.set_signals({
        sig1_time: MockSignal(score=80.0, direction="UP", entry=100, target=120, stop=80), # Wide target to keep open
        sig2_time: MockSignal(score=80.0, direction="UP", entry=100, target=120, stop=80)
    })
    
    # Price stays flat, so Trade 1 never exits
    engine = BacktestEngine(MockQuantEngine, df, "TEST")
    res = engine.run(start_idx=40)
    
    # Total trades should be 0 because Trade 1 never closed, and Trade 2 never opened
    # (Results only count closed trades)
    assert res.total_trades == 0
    # However, we want to ensure Trade 2 didn't override Trade 1 internally. 
    # Since we can't easily inspect private state without closed trades, 
    # we force an exit at index 60.
    
    # If Trade 2 overwrote Trade 1, entry time would be later.
    
    df.iloc[60, df.columns.get_loc("High")] = 130 # Hit Target
    res = engine.run(start_idx=40)
    
    assert res.total_trades == 1
    # Ensure it was the FIRST trade (entered at index 51 open)
    assert res.recent_trades[0].entry_date == df.index[51].strftime("%Y-%m-%d %H:%M")