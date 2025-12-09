# core/quant/futures_engine.py

from __future__ import annotations
import pandas as pd
import numpy as np

from .base_engine import (
    QuantResult, compute_basic_ohlc_aliases, add_core_indicators,
    compute_trend_score, detect_volatility_regime, determine_signal_quality,
    clamp_score, build_entry_target_stop
)
from .indicators import compute_all_indicators
from .ml_bridge import predict_directional_edge
from .signals.liquidity import compute_liquidity_grab_risk
from .signals.stoprun import compute_stop_run_risk


class FuturesQuantEngine:
    MARKET_TYPE = "FUTURES"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "INTRADAY") -> QuantResult:
        # 1. Prep Data
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 50:
            return QuantResult(direction="NEUTRAL", score=0, extras={"error": "Insufficient Data"})

        # 2. Add Indicators
        df = add_core_indicators(df)

        # 3. ML Prediction (The Fix)
        ml_features = compute_all_indicators(df)
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except Exception:
            raw_prob = 50.0

        ml_edge = clamp_score(raw_prob)

        # 4. Technical Logic (Velocity)
        trend_score = compute_trend_score(df)

        # Calculate Velocity (Rate of Change)
        roc = df['close'].pct_change(3).iloc[-1] * 100
        velocity_score = 0
        if abs(roc) > 0.5: velocity_score = 10
        if abs(roc) > 1.5: velocity_score = 20
        if roc < 0: velocity_score = -velocity_score

        # 5. Weighted Score
        # Futures is Momentum heavy: 50% ML, 30% Trend, 20% Velocity
        final_score = clamp_score(0.5 * ml_edge + 0.3 * (50 + trend_score) + 0.2 * (50 + velocity_score))

        # 6. Direction
        if final_score > 55:
            direction = "BULLISH"
        elif final_score < 45:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # 7. Risks
        liq_risk = compute_liquidity_grab_risk(df)
        stop_risk, _ = compute_stop_run_risk(df)

        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            trend_score=trend_score, volatility_regime=detect_volatility_regime(df),
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop,
            extras={
                "velocity": round(roc, 2),
                "liquidity_grab_risk": liq_risk,
                "stop_run_risk": stop_risk
            }
        )