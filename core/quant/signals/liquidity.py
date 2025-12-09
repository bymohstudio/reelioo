# core/quant/signals/liquidity.py

"""
Adaptive Liquidity Grab Detector (Model C)
-----------------------------------------
Institutional-grade detection for:
- Stop-hunt wicks
- Liquidity sweeps
- Trap candles
- Volatility-driven manipulation
- ATR-weighted anomalies

Works for:
- EQUITY (15m/1h)
- FUTURES (5m)
- OPTIONS (uses underlying data)
- CRYPTO (1h)

Returns:
  LOW / MEDIUM / HIGH
"""

import numpy as np

def compute_liquidity_grab_risk(df):
    """
    Core Model:
    -----------
    A liquidity grab is usually:
    - A large wick relative to candle body
    - Wick relative to ATR
    - Rapid volatility expansion
    - Break of previous swing highs/lows
    - Quick reversion (operator trap)

    This detector uses 3 candles to reduce noise.
    """

    try:
        recent = df.tail(3)

        high = recent['high'].max()
        low = recent['low'].min()
        close = recent['close'].iloc[-1]
        open_ = recent['open'].iloc[-1]

        wick_size = high - low
        body_size = abs(close - open_)

        # --- ATR (Adaptive component) ---
        # ATR from last 14 candles if available
        if len(df) >= 14:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            atr = df['tr'].rolling(14).mean().iloc[-1]
        else:
            atr = wick_size * 0.6  # fallback

        # --- Volatility spike heuristic ---
        vol = recent['close'].pct_change().std() * 100  # %
        vol_spike = vol > (recent['close'].pct_change().std() * 100) * 1.5

        # --- Swing point sweep detection ---
        # If this candle swept the last high/low but closed inside range
        prev_high = df['high'].iloc[-4]
        prev_low = df['low'].iloc[-4]

        swept_high = high > prev_high and close < high
        swept_low = low < prev_low and close > low
        sweep_detected = swept_high or swept_low

        # --- Risk Modeling ---
        wick_vs_body = wick_size / max(body_size, 1e-6)
        wick_vs_atr = wick_size / max(atr, 1e-6)

        # High Risk:
        if wick_vs_atr > 1.7 and wick_vs_body > 4 and sweep_detected:
            return "HIGH"

        if wick_vs_atr > 1.2 and vol > 1.2 and sweep_detected:
            return "HIGH"

        # Medium:
        if wick_vs_atr > 0.8 or wick_vs_body > 2:
            return "MEDIUM"

        # Low:
        return "LOW"

    except Exception:
        # If data insufficient, mark medium so user stays cautious.
        return "MEDIUM"
