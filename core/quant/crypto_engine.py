# core/quant/crypto_engine.py

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


class CryptoQuantEngine:
    MARKET_TYPE = "CRYPTO"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "SWING") -> QuantResult:
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 50:
            return QuantResult(direction="NEUTRAL", score=0, extras={"error": "Insufficient Data"})

        df = add_core_indicators(df)

        # 1. ML
        ml_features = compute_all_indicators(df)
        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except Exception:
            raw_prob = 50.0
        ml_edge = clamp_score(raw_prob)

        # 2. Market Phase (Accumulation vs Expansion)
        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)

        # ATR Squeeze logic
        atr = df['atr_14'].iloc[-1]
        avg_atr = df['atr_14'].rolling(50).mean().iloc[-1]
        phase = "NORMAL"
        if atr < avg_atr * 0.7:
            phase = "SQUEEZE (PRE-BREAKOUT)"
        elif atr > avg_atr * 1.3:
            phase = "EXPANSION (RUN)"

        # 3. Score
        final_score = clamp_score(0.65 * ml_edge + 0.35 * (50 + trend_score))

        direction = "NEUTRAL"
        if final_score > 53:
            direction = "BUY"
        elif final_score < 47:
            direction = "SELL"

        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style, atr_fallback=2.0)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop,
            extras={
                "market_phase": phase
            }
        )