# core/quant/quant_engine.py

from __future__ import annotations

from typing import Type

from .base_engine import QuantResult
from .equity_engine import EquityQuantEngine
from .futures_engine import FuturesQuantEngine
from .options_engine import OptionsQuantEngine
from .crypto_engine import CryptoQuantEngine


ENGINE_MAP = {
    "EQUITY": EquityQuantEngine,
    "FUTURES": FuturesQuantEngine,
    "OPTIONS": OptionsQuantEngine,
    "CRYPTO": CryptoQuantEngine,
}


def get_engine_cls(market_type: str) -> Type:
    return ENGINE_MAP.get(market_type.upper(), EquityQuantEngine)


def run_quant(
    market_type: str,
    df,
    symbol: str,
    trade_style: str = "SWING",
) -> QuantResult:
    engine = get_engine_cls(market_type)
    return engine.run(df, symbol, trade_style)
