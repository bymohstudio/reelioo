# core/quant/quant_engine.py

import pandas as pd
from .indicators import prepare_indicators

# -------------------------------------------
# CONFIG: STYLE + RISK PROFILES
# -------------------------------------------

STYLE_CONFIG = {
    "INTRADAY": {
        "target_atr": 1.2,
        "stop_atr": 0.5,
        "time_frame": "3–6 Hours",
    },
    "SWING": {
        "target_atr": 2.0,
        "stop_atr": 1.0,
        "time_frame": "2–5 Days",
    },
    "LONG_TERM": {
        "target_atr": 4.0,
        "stop_atr": 2.0,
        "time_frame": "3–8 Weeks",
    },
}

DEFAULT_STYLE = "SWING"

def _safe_last(series, default=0.0):
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return default

def _pick_style(trade_style: str | None) -> str:
    if not trade_style: return DEFAULT_STYLE
    s = trade_style.strip().upper()
    if s in ("INTRADAY", "SCALP", "DAY"): return "INTRADAY"
    if s in ("LONG", "LONGTERM", "POSITION", "LONG_TERM"): return "LONG_TERM"
    return "SWING"

def _score_trend(df: pd.DataFrame) -> float:
    close = _safe_last(df["Close"])
    ma20 = _safe_last(df["ma20"])
    ma50 = _safe_last(df["ma50"])
    
    if ma20 == 0 or ma50 == 0: return 50.0

    score = 50.0
    if close > ma20 > ma50: score = 90.0
    elif close > ma20 and ma20 < ma50: score = 65.0
    elif close < ma20 < ma50: score = 10.0
    elif close < ma20 and ma20 > ma50: score = 35.0
    return score

def _score_momentum(df: pd.DataFrame) -> float:
    rsi = _safe_last(df["rsi"], 50.0)
    if rsi > 70: return 40.0 # Caution overbought
    if rsi < 30: return 60.0 # Oversold bounce
    if 50 <= rsi <= 70: return 85.0 # Strong Bull
    return 30.0

def _classify_regime(df: pd.DataFrame) -> str:
    width = _safe_last(df.get("bb_width"), 0.0)
    # Simple relative check - if width > recent avg
    if width > 0.05: return "High Volatility"
    return "Range-Bound"

def run_signal_engine(
    df: pd.DataFrame,
    symbol: str,
    market_type: str = "GLOBAL",
    trade_style: str | None = None,
) -> dict:
    
    if df is None or df.empty:
        return {"error": "No Data Available"}

    style_key = _pick_style(trade_style)
    
    # 1. Prepare Indicators
    df = prepare_indicators(df, asset_type=market_type)
    
    # 2. Get latest values
    close = _safe_last(df["Close"])
    atr = _safe_last(df["atr"])
    
    if close == 0: return {"error": "Invalid Price Data"}

    # 3. Calculate Component Scores
    trend = _score_trend(df)
    mom = _score_momentum(df)
    vol = 50.0 # Placeholder vol score
    
    # Weighted Composite Score
    composite = (trend * 0.4) + (mom * 0.4) + (vol * 0.2)
    composite = max(0, min(100, int(composite)))
    
    # 4. Determine Direction & Label
    if composite >= 75:
        direction = "UP"
        label = "STRONG BUY"
    elif composite >= 60:
        direction = "UP"
        label = "BUY"
    elif composite <= 25:
        direction = "DOWN"
        label = "AVOID"
    elif composite <= 40:
        direction = "DOWN"
        label = "SELL"
    else:
        direction = "SIDEWAYS"
        label = "NEUTRAL"

    # 5. Targets (ATR Based)
    cfg = STYLE_CONFIG[style_key]
    
    if direction == "UP":
        entry = close
        stop = close - (atr * cfg["stop_atr"])
        target = close + (atr * cfg["target_atr"])
    elif direction == "DOWN":
        entry = close
        stop = close + (atr * cfg["stop_atr"])
        target = close - (atr * cfg["target_atr"])
    else:
        # Neutral bounds
        entry = close
        stop = close * 0.98
        target = close * 1.02

    # 6. S/R Levels
    recent_high = float(df["High"].tail(20).max())
    recent_low = float(df["Low"].tail(20).min())

    return {
        "symbol": symbol.upper(),
        "market_type": market_type,
        "score": composite,
        "direction": direction,
        "label": label,
        "close": round(close, 2),
        "targets": {
            "entry": round(entry, 2),
            "target": round(target, 2),
            "stop_loss": round(stop, 2)
        },
        "levels": {
            "support": round(recent_low, 2),
            "resistance": round(recent_high, 2)
        },
        "prediction": {
            "time_frame": cfg["time_frame"],
            "regime": _classify_regime(df),
            "guidance": f"Market is {_classify_regime(df)}."
        },
        "factors": {
            "trend_score": trend,
            "momentum_score": mom,
            "volatility_score": vol
        }
    }