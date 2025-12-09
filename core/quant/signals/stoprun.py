# core/quant/signals/stoprun.py

"""
Adaptive Stop-Run Risk Detector (Model C)
-----------------------------------------
Designed to detect stop-run / liquidity-sweep events across markets.

Inputs:
    df: pandas.DataFrame with columns ['open','high','low','close','volume'] indexed by timestamp.
         Ideally contains >= 30 rows; function will fallback gracefully with fewer rows.

Outputs:
    - risk_level: "LOW" / "MEDIUM" / "HIGH"
    - details (optional): dict with metrics used (wick_vs_atr, vol_z, body_pct, sweep_flag)

Notes:
    - Adaptive thresholds use ATR, vol z-score, and wick/body ratios.
    - Optimized for intraday TFs (5m, 15m, 1h) but will work on others.
"""

import numpy as np

def _safe_rolling_atr(df, period=14):
    """
    Compute ATR safely; returns last ATR value or None.
    """
    try:
        if len(df) < 2:
            return None
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = np.maximum.reduce([tr1.fillna(0).values, tr2.fillna(0).values, tr3.fillna(0).values])
        if len(tr) < period:
            # fallback: simple mean of available TRs
            return float(np.nanmean(tr[-period:])) if len(tr) > 0 else None
        atr = float(np.nanmean(tr[-period:]))
        return atr
    except Exception:
        return None


def _vol_z_score(df, lookback=50):
    """
    Compute z-score of recent volume vs historical window.
    Returns last volume zscore (float).
    """
    try:
        if 'volume' not in df.columns:
            return 0.0
        vols = df['volume'].fillna(0).values
        if len(vols) < 5:
            return 0.0
        window = min(len(vols), lookback)
        hist = vols[-window:-1] if len(vols) > 1 else vols
        if len(hist) < 3:
            mean = np.mean(hist) if len(hist) > 0 else 0.0
            std = np.std(hist) if len(hist) > 0 else 0.0
        else:
            mean = np.mean(hist)
            std = np.std(hist, ddof=0)
        last_vol = vols[-1]
        if std <= 1e-9:
            return 0.0
        z = (last_vol - mean) / std
        return float(z)
    except Exception:
        return 0.0


def _detect_sweep(df):
    """
    Detect sweep: last candle pierces previous swing high/low then closes back inside.
    Returns True if sweep-like pattern detected.
    """
    try:
        if len(df) < 5:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        window_high = df['high'].iloc[-5:-1].max() if len(df) >= 6 else df['high'].iloc[:-1].max()
        window_low = df['low'].iloc[-5:-1].min() if len(df) >= 6 else df['low'].iloc[:-1].min()

        # Up-sweep: last high exceeded previous window high, but closed back below it
        up_sweep = (last['high'] > window_high) and (last['close'] < last['high']) and (last['close'] < window_high)
        # Down-sweep: last low below previous window low, but closed back above it
        down_sweep = (last['low'] < window_low) and (last['close'] > last['low']) and (last['close'] > window_low)

        return up_sweep or down_sweep
    except Exception:
        return False


def compute_stop_run_risk(df):
    """
    Main entrypoint.
    Returns: risk_level (str), details (dict)
    """
    try:
        # Basic safety checks
        if df is None or len(df) < 2:
            return "MEDIUM", {"reason": "insufficient_data"}

        recent = df.tail(3)
        last = recent.iloc[-1]
        prev = recent.iloc[-2]

        high = float(recent['high'].max())
        low = float(recent['low'].min())
        close = float(last['close'])
        open_ = float(last['open'])
        body = abs(close - open_)
        wick_total = high - low

        # ATR adaptive baseline
        atr = _safe_rolling_atr(df, period=14) or max( (wick_total * 0.5), 1e-6)

        # wick vs atr and body
        wick_vs_atr = wick_total / max(atr, 1e-9)
        body_pct = (body / max(wick_total, 1e-9)) * 100

        # volume z-score
        vol_z = _vol_z_score(df, lookback=50)

        # sweep detection
        sweep_flag = _detect_sweep(df)

        # Price reversion quickness: did price recover inside range after sweep?
        # Measure close relative to intraday extremes
        recovered_percent = None
        try:
            if sweep_flag:
                # how far back inside the sweep the close is
                if last['high'] - last['low'] > 0:
                    recovered_percent = (abs(last['close'] - ( (last['high']+last['low'])/2)) / (last['high'] - last['low'])) * 100
        except Exception:
            recovered_percent = None

        # Adaptive thresholds by market type heuristics (we don't have market type here,
        # so use generic adaptive logic based on ATR & vol_z)
        # High risk conditions:
        # - wick significantly larger than ATR
        # - body very small compared to wick (indicates rejection)
        # - volume spike (z > 2)
        # - sweep detected
        high_risk = False
        if (wick_vs_atr > 1.5 and body_pct < 25 and vol_z > 2 and sweep_flag):
            high_risk = True
        elif (wick_vs_atr > 2.2 and body_pct < 40 and (vol_z > 1.5 or sweep_flag)):
            high_risk = True

        # Medium risk heuristics:
        medium_risk = False
        if (wick_vs_atr > 1.0 and body_pct < 40 and (vol_z > 1.0 or sweep_flag)):
            medium_risk = True
        elif (wick_vs_atr > 0.8 and vol_z > 1.5):
            medium_risk = True

        # Final decision
        if high_risk:
            lvl = "HIGH"
        elif medium_risk:
            lvl = "MEDIUM"
        else:
            lvl = "LOW"

        details = {
            "wick_vs_atr": round(float(wick_vs_atr), 3),
            "body_pct": round(float(body_pct), 2),
            "vol_z": round(float(vol_z), 3),
            "sweep_flag": bool(sweep_flag),
            "recovered_percent": round(float(recovered_percent), 2) if recovered_percent is not None else None,
            "atr": float(atr)
        }

        return lvl, details

    except Exception:
        return "MEDIUM", {"reason": "exception"}
