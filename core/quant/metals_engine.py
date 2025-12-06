from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_engine import (
    SignalResult,
    STYLE_CONFIG,
    _safe_last,
    _pick_style,
    compute_basic_ohlc_aliases,
    compute_atr,
    compute_rsi,
    compute_vol_regime,
    expected_bars_to_target,
    format_time_window,
    Direction,
)

class MetalsQuantEngine:
    """
    Specialized engine for Gold (XAU), Silver (XAG), and other precious metals.
    Focuses on:
    1. Macro Trend Alignment (MA50 vs MA200)
    2. Volatility Compression (Squeeze before breakout)
    3. Structural Breakouts (Recent Highs)
    """
    MODEL_VERSION = "metals_v2.0"

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_basic_ohlc_aliases(df)
        close = df["Close"]
        
        # Metals respect long-term levels
        df["ma20"] = close.rolling(20).mean()
        df["ma50"] = close.rolling(50).mean()
        df["ma200"] = close.rolling(200).mean()
        
        df["atr14"] = compute_atr(df, 14)
        df["rsi14"] = compute_rsi(close, 14)
        
        # Volatility Compression Metric: ATR / Price
        df["vol_ratio"] = df["atr14"] / close
        
        return df

    @classmethod
    def _trend_score(cls, df: pd.DataFrame) -> float:
        c = _safe_last(df["Close"], 0)
        m20 = _safe_last(df["ma20"], 0)
        m50 = _safe_last(df["ma50"], 0)
        m200 = _safe_last(df["ma200"], 0)
        
        score = 50.0
        
        # Gold/Silver 'Golden Cross' Alignment
        if m50 > m200:
            score += 10.0
            if c > m50: score += 10.0
            if c > m20: score += 10.0
        elif m50 < m200:
            score -= 10.0
            if c < m50: score -= 10.0
            if c < m20: score -= 10.0
            
        # Immediate Price Action
        if c > m20 and c > m50 and c > m200:
            # Full Bull Alignment
            return 90.0
        elif c < m20 and c < m50 and c < m200:
            # Full Bear Alignment
            return 10.0
            
        return float(np.clip(score, 0, 100))

    @classmethod
    def _momentum_score(cls, df: pd.DataFrame) -> float:
        rsi = _safe_last(df["rsi14"], 50)
        
        # Metals Momentum Zones
        if 55 <= rsi <= 70:
            # Strong Bullish Momentum (Gold trends well here)
            return 80.0
        elif 40 <= rsi < 55:
            # Consolidation / Chop
            return 50.0
        elif rsi > 75:
            # Overbought - Risk of pullback in metals is high
            return 40.0
        elif rsi < 30:
            # Oversold - Metals often snap back hard
            return 60.0 # Counter-trend bounce potential
        elif rsi < 40:
            # Bearish drag
            return 30.0
            
        return 50.0

    @classmethod
    def _squeeze_score(cls, df: pd.DataFrame) -> float:
        """
        Detects volatility compression. 
        Metals often explode after a period of low ATR (Squeeze).
        """
        vol_ratio = df["vol_ratio"].tail(20) # Last 20 bars
        current_vol = _safe_last(df["vol_ratio"])
        avg_vol = vol_ratio.mean()
        
        # If current vol is significantly lower than average, we are in a squeeze
        if current_vol < avg_vol * 0.8:
            # Compression -> High potential energy
            # We treat this as a neutral-to-positive setup multiplier
            return 70.0 
        elif current_vol > avg_vol * 1.5:
            # Expansion -> Move already happened or happening
            return 50.0
            
        return 50.0

    @classmethod
    def _structure_score(cls, df: pd.DataFrame) -> float:
        # Breakout check
        recent = df.tail(20)
        high_max = recent["High"].max()
        low_min = recent["Low"].min()
        close = _safe_last(df["Close"])
        
        # Near Highs?
        range_size = high_max - low_min
        if range_size == 0: return 50.0
        
        pos_in_range = (close - low_min) / range_size
        
        if pos_in_range > 0.85: return 85.0 # Breakout imminent
        if pos_in_range < 0.15: return 15.0 # Breakdown imminent
        
        return 50.0

    @classmethod
    def _composite(cls, t, m, sq, st) -> float:
        # Trend (35%), Momentum (30%), Structure (20%), Squeeze (15%)
        return float(np.clip(t * 0.35 + m * 0.30 + st * 0.20 + sq * 0.15, 0, 100))

    @classmethod
    def _dir_label(cls, comp: float) -> tuple[Direction, str]:
        if comp >= 80:
            return "UP", "STRONG ACCUMULATION"
        if comp >= 65:
            return "UP", "BULLISH BIAS"
        if comp <= 20:
            return "DOWN", "STRONG DISTRIBUTION"
        if comp <= 40:
            return "DOWN", "BEARISH BIAS"
        return "FLAT", "CONSOLIDATION"

    @classmethod
    def run(cls, df: pd.DataFrame, symbol: str, trade_style: str | None = None) -> SignalResult:
        if df is None or df.empty:
            raise ValueError("Metals engine: empty dataframe")
            
        style = _pick_style(trade_style)
        df = cls._prepare(df)
        close = _safe_last(df["Close"], 0.0)
        
        if close is None or close <= 0:
            raise ValueError("Metals engine: invalid close")
            
        # Component Scores
        t = cls._trend_score(df)
        m = cls._momentum_score(df)
        sq = cls._squeeze_score(df)
        st = cls._structure_score(df)
        
        # Composite
        comp = cls._composite(t, m, sq, st)
        direction, label = cls._dir_label(comp)
        
        # Support/Resistance (Crucial for metals)
        support = float(df.tail(50)["Low"].min())
        resistance = float(df.tail(50)["High"].max())
        
        # Volatility Regime for targets
        atr = _safe_last(df["atr14"], close * 0.01)
        atr_pct = atr / close
        regime = compute_vol_regime(atr_pct)
        
        # Dynamic Targeting
        cfg = STYLE_CONFIG[style]
        target_mult = cfg["target_atr"]
        stop_mult = cfg["stop_atr"]
        
        # Metals often extend further in trends
        if regime == "HIGH_ENERGY_BREAKOUT":
            target_mult *= 1.2
            
        if direction == "UP":
            entry = close
            target = close + (atr * target_mult)
            stop = close - (atr * stop_mult)
        elif direction == "DOWN":
            entry = close
            target = close - (atr * target_mult)
            stop = close + (atr * stop_mult)
        else:
            entry = close
            target = close * 1.005
            stop = close * 0.995

        dist = abs(target - entry)
        bars, hrs = expected_bars_to_target(dist, atr, cfg["bar_hours"])
        window = format_time_window(hrs)

        factors: Dict[str, float] = {
            "trend_score": round(t, 1),
            "momentum_score": round(m, 1),
            "structure_score": round(st, 1),
            "squeeze_score": round(sq, 1),
            "composite_score": round(comp, 1),
        }
        
        meta: Dict[str, Any] = {
            "model_version": cls.MODEL_VERSION,
            "vol_regime": regime,
            "atr_14": round(atr, 4)
        }

        return SignalResult(
            symbol=symbol.upper(),
            market="METALS",
            style=style,
            model_version=cls.MODEL_VERSION,
            score=factors["composite_score"],
            direction=direction,
            label=label,
            entry=round(entry, 2),
            target=round(target, 2),
            stop=round(stop, 2),
            support=round(support, 2),
            resistance=round(resistance, 2),
            time_frame=window,
            expected_bars_to_target=bars,
            expected_time_to_target_hours=hrs,
            factors=factors,
            meta=meta,
        )