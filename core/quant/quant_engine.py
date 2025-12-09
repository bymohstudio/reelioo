# core/quant/quant_engine.py

from __future__ import annotations
import logging
from .base_engine import QuantResult, compute_basic_ohlc_aliases

# Import Engines
from .equity_engine import EquityQuantEngine
from .futures_engine import FuturesQuantEngine
from .options_engine import OptionsQuantEngine
from .crypto_engine import CryptoQuantEngine

log = logging.getLogger(__name__)

ENGINE_MAP = {
    "EQUITY": EquityQuantEngine,
    "FUTURES": FuturesQuantEngine,
    "OPTIONS": OptionsQuantEngine,
    "CRYPTO": CryptoQuantEngine
}


def _get_engine_cls(market_type: str):
    mt = (market_type or "EQUITY").strip().upper()
    return ENGINE_MAP.get(mt, EquityQuantEngine)


def run_quant(market_type, df, symbol, trade_style="SWING") -> QuantResult:
    """
    Main Orchestrator.
    """
    try:
        # Global Data Pre-check
        if df is None or df.empty:
            return QuantResult(score=0, direction="NEUTRAL", extras={"error": "Empty Dataframe"})

        # Route to specific engine
        engine_cls = _get_engine_cls(market_type)
        result = engine_cls.run(df, symbol, trade_style)

        # Safety Clamp
        if result.score < 0: result.score = 0
        if result.score > 100: result.score = 100

        return result

    except Exception as e:
        log.exception(f"Quant Engine Crash: {e}")
        return QuantResult(direction="NEUTRAL", score=0, extras={"error": str(e)})