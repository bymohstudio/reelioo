# core/quant/options_engine.py

from __future__ import annotations
import pandas as pd
from .base_engine import (
    QuantResult, compute_basic_ohlc_aliases, add_core_indicators,
    compute_trend_score, detect_volatility_regime, determine_signal_quality,
    clamp_score, build_entry_target_stop
)
from .indicators import compute_all_indicators
from .ml_bridge import predict_directional_edge


class OptionsQuantEngine:
    """
    Institutional Options Engine.
    Fixes: ML Feature Mismatch.
    Adds: Dynamic Strategy Selection (Buy vs Sell) based on Volatility.
    """
    MARKET_TYPE = "OPTIONS"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "INTRADAY") -> QuantResult:
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 80: return cls._empty_result(symbol, trade_style)

        df = add_core_indicators(df)
        last_price = float(df.iloc[-1]["close"])

        # 1. ML Fix
        ml_features = compute_all_indicators(df)
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except:
            raw_prob = 50.0
        ml_edge = clamp_score(raw_prob)

        # 2. Context
        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)

        # 3. Scoring
        # Penalize buying in high chop
        penalty = 10 if vol_regime == "HIGH" else 0
        final_score = clamp_score((0.7 * ml_edge + 0.3 * (50 + trend_score / 2)) - penalty)

        # 4. Strategy Logic
        if final_score > 55:
            direction = "BUY"  # Bullish Underlying
            opt_type = "CE"
        elif final_score < 45:
            direction = "SELL"  # Bearish Underlying
            opt_type = "PE"
        else:
            direction = "NEUTRAL"
            opt_type = "-"

        # 5. Smart Instrument Selection
        # ATM Strike Calculation
        strike_step = 100 if "BANK" in symbol else 50
        atm_strike = round(last_price / strike_step) * strike_step

        strategy = "WAIT"
        theta_risk = "MODERATE"

        if direction != "NEUTRAL":
            if vol_regime == "HIGH":
                # High Vol = Expensive Premiums = SELL/SPREAD
                action = "CREDIT SPREAD"
                theta_risk = "FRIENDLY (SELLER)"
            else:
                # Low Vol = Cheap Premiums = BUY
                action = "LONG (BUY)"
                theta_risk = "HIGH (BUYER)"

            strategy = f"{action} {atm_strike} {opt_type}"

        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            raw_prob=ml_edge / 100, trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop, risk_reward=rr,
            extras={
                "opt_strategy": strategy,
                "theta_risk": theta_risk
            }
        )

    @staticmethod
    def _empty_result(symbol, style):
        return QuantResult(symbol, "OPTIONS", style, "NEUTRAL", 0, 0, 0.5, 0, "UNKNOWN", "C", 0, 0, 0, 0)