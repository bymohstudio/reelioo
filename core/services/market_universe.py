# core/services/market_universe.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class MarketDefinition:
    """
    Canonical list of markets Reelioo supports.

    code   -> Used by frontend + AnalyzeMarketView (EQUITY / FUTURES / OPTIONS / CRYPTO)
    label  -> Clean UI name (what user sees)
    description -> Tooltip / subtitle explaining the market
    default_trade_style -> Suggested mode (INTRADAY / SWING / LONG_TERM)
    segment -> Internal segment tag (NSE / NFO / CRYPTO etc.)
    enabled -> Toggle without deleting the entry
    """
    code: str
    label: str
    description: str
    default_trade_style: str
    segment: str
    enabled: bool = True


# 🔹 Curated institutional universe for Indian-only Reelioo
INSTITUTIONAL_MARKETS: List[MarketDefinition] = [
    MarketDefinition(
        code="EQUITY",
        label="NSE Stocks (EQ)",
        description="Cash market stocks like RELIANCE, TCS, HDFCBANK traded on NSE.",
        default_trade_style="SWING",
        segment="NSE",
    ),
    MarketDefinition(
        code="FUTURES",
        label="Index Futures (NIFTY / BANKNIFTY)",
        description="NSE index futures for directional and hedging trades.",
        default_trade_style="INTRADAY",
        segment="NFO",
    ),
    MarketDefinition(
        code="OPTIONS",
        label="Index Options (NIFTY / BANKNIFTY)",
        description="Weekly index options (ATM / near-ATM) suitable for intraday strategies.",
        default_trade_style="INTRADAY",
        segment="NFO",
    ),
    MarketDefinition(
        code="CRYPTO",
        label="BTC-INR (Crypto)",
        description="Bitcoin vs INR sourced via Yahoo Finance for 24x7 tracking.",
        default_trade_style="SWING",
        segment="CRYPTO",
    ),
]


def get_enabled_markets() -> List[Dict]:
    """
    Returns a list of simple dicts for API/Frontend.
    This keeps your UI and backend in sync with a single source of truth.
    """
    payload: List[Dict] = []
    for m in INSTITUTIONAL_MARKETS:
        if not m.enabled:
            continue
        payload.append(
            {
                "code": m.code,
                "label": m.label,
                "description": m.description,
                "default_trade_style": m.default_trade_style,
                "segment": m.segment,
            }
        )
    return payload
