# core/quant/equity_engine.py

from __future__ import annotations
import pandas as pd
import numpy as np

from .base_engine import (
    QuantResult, compute_basic_ohlc_aliases, add_core_indicators,
    compute_trend_score, detect_volatility_regime, determine_signal_quality,
    clamp_score, build_entry_target_stop
)
# CRITICAL IMPORT: This generates the 22 features for ML
from .indicators import compute_all_indicators
from .ml_bridge import predict_directional_edge


class EquityQuantEngine:
    """
    ML-Dominant Equity Engine.
    Fixes: Now passes full feature set to ML model to prevent mismatches.
    """
    MARKET_TYPE = "EQUITY"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "SWING") -> QuantResult:
        # 1. Prep Data
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 80:
            return cls._empty_result(symbol, trade_style)

        # 2. Add Indicators
        df = add_core_indicators(df)
        last_price = float(df.iloc[-1]["close"])

        # 3. GENERATE FULL ML FEATURES (The Fix)
        # This matches the 22 features in equity_edge.json
        ml_features = compute_all_indicators(df)

        # 4. Predict
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except Exception as e:
            print(f"[ML Error] {e}")
            raw_prob = 50.0

        ml_edge = clamp_score(raw_prob)

        # 5. Technical Context
        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)

        # 6. Weighting: 70% ML, 30% Quant
        final_score = clamp_score(0.7 * ml_edge + 0.3 * (50 + trend_score / 2))

        # 7. Direction
        if ml_edge > 55:
            direction = "BUY"
        elif ml_edge < 45:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        # 8. Execution Context (Institutional Touch)
        exec_type = "INTRADAY / MIS" if trade_style == "INTRADAY" else "DELIVERY / CASH"

        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            raw_prob=ml_edge / 100, trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop, risk_reward=rr,
            extras={"exec_type": exec_type}
        )

    @staticmethod
    def _empty_result(symbol, style):
        return QuantResult(symbol, "EQUITY", style, "NEUTRAL", 0, 0, 0.5, 0, "UNKNOWN", "C", 0, 0, 0, 0)