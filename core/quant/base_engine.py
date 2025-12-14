# core/quant/base_engine.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# 1. Unified Result Object
# -------------------------------------------------------------------
@dataclass
class QuantResult:
    """
    Standardized result object used by all engines.
    """
    symbol: str
    market_type: str
    time_frame: str
    direction: str  # BUY, SELL, NEUTRAL

    # Scores
    score: float
    ml_edge: float = 0.0
    raw_prob: float = 0.5
    trend_score: float = 0.0

    # Context
    volatility_regime: str = "NORMAL"
    signal_quality: str = "NONE"

    # Trade Levels
    entry: float = 0.0
    target: float = 0.0
    stop: float = 0.0
    risk_reward: float = 0.0

    # Extra Data (Time estimates, Leverage, News, Kelly)
    extras: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------
# 2. Smart Position Sizing (Kelly Criterion)
# -------------------------------------------------------------------
class KellySizer:
    """
    Calculates Optimal Position Size using the Kelly Criterion.
    Formula: K% = W - [(1 - W) / R]
    """

    @staticmethod
    def calculate(prob_score: float, risk_reward: float, fractional: float = 0.5) -> str:
        # 1. Convert Score to Probability
        W = max(0.01, min(0.99, prob_score / 100.0))

        # 2. Risk Reward (R)
        R = max(0.1, risk_reward)

        # 3. Kelly Formula
        kelly_pct = W - ((1 - W) / R)

        # 4. Safety Scaling (Half Kelly is standard)
        safe_kelly = kelly_pct * fractional

        # 5. Constraints (Max 20% allocation per trade)
        if safe_kelly <= 0:
            return "0% (No Trade)"

        final_pct = min(0.20, safe_kelly)
        return f"{final_pct * 100:.1f}% Capital"


# -------------------------------------------------------------------
# 3. Core Math & Indicators
# -------------------------------------------------------------------
def compute_basic_ohlc_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 1. ATR (Volatility)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()

    # 2. RSI (Momentum)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 3. EMAs (Trend)
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    return df.fillna(0)


def compute_trend_score(df: pd.DataFrame) -> float:
    # Basic Trend Logic: Price vs EMAs + RSI context
    score = 0
    last = df.iloc[-1]

    # Bullish Factors
    if last['close'] > last['ema_20']: score += 20
    if last['close'] > last['ema_50']: score += 20
    if last['ema_20'] > last['ema_50']: score += 20
    if last['rsi_14'] > 50: score += 20

    # Bearish Factors
    if last['close'] < last['ema_20']: score -= 20
    if last['close'] < last['ema_50']: score -= 20
    if last['ema_20'] < last['ema_50']: score -= 20
    if last['rsi_14'] < 50: score -= 20

    return float(max(-100, min(100, score)))


def detect_volatility_regime(df: pd.DataFrame) -> str:
    current_atr = df['atr_14'].iloc[-1]
    avg_atr = df['atr_14'].rolling(50).mean().iloc[-1]

    if current_atr > avg_atr * 1.2: return "HIGH"
    if current_atr < avg_atr * 0.8: return "LOW"
    return "NORMAL"


def determine_signal_quality(score: float, ml_edge: float) -> str:
    if abs(score) > 80 and ml_edge > 55: return "HIGH_CONVICTION"
    if abs(score) > 60: return "MODERATE"
    return "WEAK"


def clamp_score(val: float) -> float:
    return max(-100, min(100, val))


# -------------------------------------------------------------------
# 4. Dynamic Level Builder
# -------------------------------------------------------------------
def build_entry_target_stop(
        df: pd.DataFrame,
        direction: str,
        trade_style: str = "SWING"
) -> Tuple[float, float, float, float]:
    """
    Calculates dynamic Entry, Target, and Stop-Loss based on Volatility (ATR).
    """
    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last.get("atr_14", 0))

    if atr <= 0: atr = price * 0.02  # Fallback 2% if ATR missing

    # Multipliers based on style
    if trade_style == "INTRADAY":
        stop_mult, tgt_mult = 1.5, 2.5
    elif trade_style == "LONG_TERM":
        stop_mult, tgt_mult = 2.5, 5.0
    else:  # SWING
        stop_mult, tgt_mult = 2.0, 3.5

    risk = stop_mult * atr
    reward = tgt_mult * atr

    if "BUY" in direction or "LONG" in direction:
        entry = price
        stop = price - risk
        target = price + reward
    elif "SELL" in direction or "SHORT" in direction:
        entry = price
        stop = price + risk
        target = price - reward
    else:
        entry = price
        stop = price
        target = price

    rr = round(reward / risk, 2) if risk > 0 else 0.0

    return round(entry, 4), round(target, 4), round(stop, 4), rr