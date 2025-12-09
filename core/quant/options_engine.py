# core/quant/options_engine.py

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

class OptionsQuantEngine:
    MARKET_TYPE = "OPTIONS"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "INTRADAY") -> QuantResult:
        # 1. Analyze Underlying
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 50:
            return QuantResult(direction="NEUTRAL", score=0, extras={"error": "Insufficient Data"})

        df = add_core_indicators(df)
        last_price = float(df.iloc[-1]["close"])

        # 2. ML Prediction
        ml_features = compute_all_indicators(df)
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except Exception:
            raw_prob = 50.0
        ml_edge = clamp_score(raw_prob)

        # 3. Technicals
        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)
        final_score = clamp_score(0.6 * ml_edge + 0.4 * (50 + trend_score))

        # 4. Direction
        direction = "NEUTRAL"
        if final_score > 55: direction = "BULLISH"
        elif final_score < 45: direction = "BEARISH"

        # 5. Smart Contract Selection
        strike_step = 100 if "BANK" in symbol else 50
        atm_strike = round(last_price / strike_step) * strike_step
        opt_type = "CE" if direction == "BULLISH" else "PE" if direction == "BEARISH" else "--"

        # 6. Manipulation Detection (Options Specific)
        gamma_pin_risk = "LOW"
        # If price is extremely close to a round number strike (within 0.1%), risk of pinning
        if abs(last_price - atm_strike) < (last_price * 0.001):
            gamma_pin_risk = "HIGH - PRICE MAGNET"

        iv_crush_risk = "LOW"
        # If Vol is HIGH but Trend is Neutral -> IV Crush Likely
        if vol_regime == "HIGH" and abs(trend_score) < 10:
            iv_crush_risk = "HIGH - PREMIUM DECAY WARNING"

        theta_risk = "HIGH (BUYER)" if trade_style == "INTRADAY" else "MODERATE"

        # 7. Premium Estimation
        target_premium = 0.0
        estimated_premium = 0.0
        if direction != "NEUTRAL":
            und_entry, und_target, _, _ = build_entry_target_stop(df, direction, trade_style)
            expected_move = abs(und_target - und_entry)
            estimated_premium = (last_price * 0.015) # Approx ATM
            target_premium = estimated_premium + (expected_move * 0.5) # Delta 0.5

        # Strategy
        strategy = f"LONG {atm_strike} {opt_type}"
        if vol_regime == "HIGH":
            strategy = f"CREDIT SPREAD (SELL {atm_strike} { 'PE' if direction=='BULLISH' else 'CE'})"

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=0.0, target=0.0, stop=0.0,
            extras={
                "opt_strategy": strategy,
                "strike": atm_strike,
                "ltp_info": {"ltp": f"{estimated_premium:.2f}"},
                "target_premium": f"{target_premium:.2f}",
                "theta_risk": theta_risk,
                "gamma_pin_risk": gamma_pin_risk,
                "iv_crush_risk": iv_crush_risk,
                "underlying_price": last_price
            }
        )