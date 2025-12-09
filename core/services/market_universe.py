# core/services/market_universe.py

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MarketDefinition:
    code: str
    label: str
    description: str
    default_trade_style: str
    segment: str
    enabled: bool = True

INSTITUTIONAL_MARKETS: List[MarketDefinition] = [
    MarketDefinition(
        code="EQUITY",
        label="NSE Stocks (EQ)",
        description="Cash market stocks like RELIANCE, TCS, HDFCBANK traded on NSE.",
        default_trade_style="SWING",
        segment="NSE"
    ),
    MarketDefinition(
        code="FUTURES",
        label="Index Futures (NIFTY / BANKNIFTY)",
        description="NSE index futures for directional and hedging trades.",
        default_trade_style="INTRADAY",
        segment="NFO"
    ),
    MarketDefinition(
        code="OPTIONS",
        label="Index Options (NIFTY / BANKNIFTY)",
        description="Weekly index options (ATM / near-ATM).",
        default_trade_style="INTRADAY",
        segment="NFO"
    ),
    MarketDefinition(
        code="CRYPTO",
        label="BTC-INR (Crypto)",
        description="Bitcoin INR price via Yahoo Finance fallback.",
        default_trade_style="SWING",
        segment="CRYPTO"
    ),
]

def get_enabled_markets() -> List[Dict]:
    payload: List[Dict] = []
    for m in INSTITUTIONAL_MARKETS:
        if not m.enabled:
            continue

        payload.append({
            "code": m.code,
            "label": m.label,
            "description": m.description,
            "default_trade_style": m.default_trade_style,
            "segment": m.segment
        })

    return payload
