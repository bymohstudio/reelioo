# core/quant/options_helpers.py
from datetime import datetime
from typing import Dict, Any


def pick_expiry_from_symbol_info(symbol_info: Dict[str, Any]) -> str | None:
    """
    Try to extract expiry from instrument metadata returned by search_service.
    symbol_info is expected to be a dict like {"symbol": "...", "expiry": "YYYY-MM-DD", ...}
    """
    try:
        exp = symbol_info.get("expiry")
        if exp:
            # Normalize to ISO date
            return str(exp)
    except Exception:
        pass
    return None


def choose_nearest_strike_from_chain(option_chain: Dict[str, Any], underlying_price: float):
    """
    option_chain: { "expiries": ["2025-12-11", ...], "chains": { "2025-12-11": [ {"strike":25950, ...}, ... ] } }
    returns dict with expiry, strike, strike_step, opt_type
    """
    if not option_chain:
        # fallback rounding
        step = 100 if underlying_price > 1000 else 50
        atm = round(underlying_price / step) * step
        return {"expiry": None, "strike": int(atm), "opt_type": "PE", "strike_step": step}

    expiries = option_chain.get("expiries", [])
    if expiries:
        chosen = expiries[0]
    else:
        chosen = None

    chains = option_chain.get("chains", {}) or {}
    if chosen and chains.get(chosen):
        strikes = sorted({int(x.get("strike")) for x in chains.get(chosen) if x.get("strike")})
        if strikes:
            nearest = min(strikes, key=lambda s: abs(s - underlying_price))
            step = strikes[1] - strikes[0] if len(strikes) > 1 else (100 if underlying_price > 1000 else 50)
            return {"expiry": chosen, "strike": int(nearest), "opt_type": "PE", "strike_step": step}

    # fallback ATM
    step = 100 if underlying_price > 1000 else 50
    atm = round(underlying_price / step) * step
    return {"expiry": chosen, "strike": int(atm), "opt_type": "PE", "strike_step": step}
