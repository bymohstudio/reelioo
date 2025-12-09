# core/quant/equity_engine.py

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


class EquityQuantEngine:
    MARKET_TYPE = "EQUITY"

    @classmethod
    def run(cls, raw_df: pd.DataFrame, symbol: str, trade_style: str = "SWING") -> QuantResult:
        # 1. Prep Data
        df = compute_basic_ohlc_aliases(raw_df)
        if df.empty or len(df) < 50:
            return QuantResult(direction="NEUTRAL", score=0, extras={"error": "Insufficient Data"})

        df = add_core_indicators(df)

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
        if final_score > 55:
            direction = "BULLISH"
        elif final_score < 45:
            direction = "BEARISH"

        # 5. Manipulation Signals
        liq_risk = compute_liquidity_grab_risk(df)
        stop_risk, _ = compute_stop_run_risk(df)

        # Whale Volume Proxy (Z-Score > 3)
        vol_z = 0
        if "volume" in df.columns:
            last_vol = df['volume'].iloc[-1]
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            std_vol = df['volume'].rolling(20).std().iloc[-1]
            if std_vol > 0:
                vol_z = (last_vol - avg_vol) / std_vol

        entry, target, stop, rr = build_entry_target_stop(df, direction, trade_style)

        return QuantResult(
            symbol=symbol, market_type=cls.MARKET_TYPE, time_frame=trade_style,
            direction=direction, score=final_score, ml_edge=ml_edge,
            trend_score=trend_score, volatility_regime=vol_regime,
            signal_quality=determine_signal_quality(final_score, ml_edge),
            entry=entry, target=target, stop=stop,
            extras={
                "liquidity_grab_risk": liq_risk,
                "stop_run_risk": stop_risk,
                "whale_activity_z": vol_z
            }
        )