# core/quant/crypto_engine.py

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


class CryptoQuantEngine:
    """
    Institutional Crypto Engine (BTC/ETH).
    Focuses on 24/7 Market Phases and Volatility Expansion.
    """
    MARKET_TYPE = "CRYPTO"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "SWING") -> QuantResult:
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 80: return cls._empty_result(symbol, trade_style)

        df = add_core_indicators(df)
        ml_features = compute_all_indicators(df)

        try:
            raw_prob = float(predict_directional_edge(cls.MARKET_TYPE, symbol, trade_style, ml_features))
        except:
            raw_prob = 50.0
        ml_edge = clamp_score(raw_prob)

        trend_score = compute_trend_score(df)
        vol_regime = detect_volatility_regime(df)

        # 3. Institutional Crypto Logic
        # Identify Market Phase using ATR compression
        atr = df['atr_14'].iloc[-1]
        avg_atr = df['atr_14'].rolling(50).mean().iloc[-1]

        phase = "NORMAL"
        if atr < avg_atr * 0.7:
            phase = "SQUEEZE (ACCUMULATION)"
        elif atr > avg_atr * 1.3:
            phase = "EXPANSION (RUN)"

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
                "market_phase": phase,
                "session": "24/7 GLOBAL",
                "vol_note": "High Variance"
            }
        )

    @staticmethod
    def _empty_result(symbol, style):
        return QuantResult(symbol, "CRYPTO", style, "NEUTRAL", 0, 0, 0.5, 0, "UNKNOWN", "C", 0, 0, 0, 0)