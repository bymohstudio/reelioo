# core/quant/signals/option_contract_parser.py

import re
from datetime import datetime
from typing import Optional, Dict

# Supported underlyings
UNDERLYINGS = [
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
]

# Month abbreviation mapping
MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def detect_underlying(symbol: str) -> Optional[str]:
    """
    Detects which underlying the option contract belongs to.
    Example:
      NIFTY25DEC22500CE -> NIFTY
      FINNIFTY02JAN24000PE -> FINNIFTY
    """
    for u in UNDERLYINGS:
        if symbol.startswith(u):
            return u
    return None


def extract_expiry(symbol: str, underlying: str) -> Optional[str]:
    """
    Extract expiry from the remainder of the symbol after removing the underlying.
    Example:
      Symbol: NIFTY25DEC22500CE
      Underlying: NIFTY
      Remainder: 25DEC22500CE

    Expected expiry: 25DEC or 02JAN or 27FEB etc.
    """
    remainder = symbol[len(underlying):]

    # Expiry format: DDMMM
    # Examples: 25DEC, 02JAN, 27FEB
    match = re.match(r"^(\d{2})([A-Z]{3})", remainder)
    if not match:
        return None

    day = match.group(1)
    mon = match.group(2)

    if mon not in MONTHS:
        return None

    # Year inference:
    # If today is late in the year and expiry month < current month -> next year
    current_year = datetime.utcnow().year
    mon_num = int(MONTHS[mon])
    current_month = datetime.utcnow().month

    if mon_num < current_month:
        year = current_year + 1
    else:
        year = current_year

    # Build final expiry date
    try:
        expiry_str = f"{year}-{MONTHS[mon]}-{day}"
        datetime.strptime(expiry_str, "%Y-%m-%d")  # validate
        return expiry_str
    except Exception:
        return None


def extract_strike_and_type(symbol: str, underlying: str) -> Dict:
    """
    After removing underlying + expiry portion, the strike + type pattern is usually:
      <strike><CE/PE>

    Example:
      FINNIFTY02JAN24000CE -> strike=24000, type=CE
      MIDCPNIFTY09JAN16500PE -> strike=16500, type=PE
    """
    remainder = symbol[len(underlying):]

    # Remove expiry part (DDMMM = 5 chars)
    core = remainder[5:]

    # Extract CE/PE
    opt_type = None
    if core.endswith("CE"):
        opt_type = "CE"
        strike_part = core[:-2]
    elif core.endswith("PE"):
        opt_type = "PE"
        strike_part = core[:-2]
    else:
        strike_part = core

    # Strike must be numeric
    strike = None
    try:
        strike = int(strike_part)
    except Exception:
        pass

    return {
        "strike": strike,
        "option_type": opt_type
    }


def parse_option_symbol(symbol: str) -> Dict:
    """
    FINAL WRAPPER:
    Returns structured data for any supported OPTIONS contract.
    """
    underlying = detect_underlying(symbol)

    if not underlying:
        return {
            "underlying": None,
            "expiry": None,
            "strike": None,
            "option_type": None,
            "valid": False
        }

    expiry = extract_expiry(symbol, underlying)
    core = extract_strike_and_type(symbol, underlying)

    return {
        "underlying": underlying,
        "expiry": expiry,
        "strike": core["strike"],
        "option_type": core["option_type"],
        "valid": underlying is not None and expiry is not None and core["strike"] is not None
    }
