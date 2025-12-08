# core/quant/futures_engine.py

from __future__ import annotations
import numpy as np
import pandas as pd
from .base_engine import (
    QuantResult, compute_basic_ohlc_aliases, add_core_indicators,
    compute_trend_score, detect_volatility_regime, determine_signal_quality,
    clamp_score, build_entry_target_stop
)
from .indicators import compute_all_indicators
from .ml_bridge import predict_directional_edge


class FuturesQuantEngine:
    """
    Institutional Futures Engine (NIFTY/BANKNIFTY).
    Focuses on Momentum Velocity and Trend Strength.
    """
    MARKET_TYPE = "FUTURES"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "INTRADAY") -> QuantResult:
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 80: return cls._empty_result(symbol, trade_style)

        df = add_core_indicators(df)
        ml_features = compute_all_indicators(df)

        raw_prob = 50.0
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except:
            pass
        ml_edge = clamp_score(raw_prob)

        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)

        # 3. Institutional Futures Logic
        # Calculate "Velocity" (Rate of Change)
        velocity = "NEUTRAL"
        roc = df['close'].pct_change(3).iloc[-1] * 100
        if abs(roc) > 1.5:
            velocity = "HIGH VELOCITY"
        elif abs(roc) > 0.5:
            velocity = "MODERATE"
        else:
            velocity = "GRINDING"

        final_score = clamp_score(0.7 * ml_edge + 0.3 * (50 + trend_score / 2))

        direction = "BUY" if ml_edge >= 52 else "SELL" if ml_edge <= 48 else "NEUTRAL"
        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            raw_prob=ml_edge / 100, trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop, risk_reward=rr,
            extras={
                "velocity": velocity,
                "contract_type": "NEAR MONTH",
                "risk_profile": "High Leverage"
            }
        )

    @staticmethod
    def _empty_result(symbol, style):
        return QuantResult(symbol, "FUTURES", style, "NEUTRAL", 0, 0, 0.5, 0, "UNKNOWN", "C", 0, 0, 0, 0)