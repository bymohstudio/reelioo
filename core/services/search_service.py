# core/services/search_service.py

import pandas as pd
import os
from django.conf import settings
from typing import List, Dict

INSTRUMENTS_DF = None


def load_instruments():
    """
    Loads & cleans SmartAPI instruments.csv
    Fixes:
    - NaN removal
    - Incorrect instrument types
    - Index/ETF filtering for equities
    - search_text normalization
    """
    global INSTRUMENTS_DF
    csv_path = os.path.join(settings.BASE_DIR, "instruments.csv")

    if not os.path.exists(csv_path):
        print("ERROR: instruments.csv not found!")
        INSTRUMENTS_DF = pd.DataFrame()
        return

    try:
        cols = ["token", "symbol", "name", "exch_seg", "instrumenttype", "expiry"]
        # Force dtype=str to prevent mixed types
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols, dtype=str)

        # CRITICAL FIX: Fill NaN with empty string to prevent boolean indexing crashes
        df = df.fillna("")

        # Filter only valid segments
        df = df[df["exch_seg"].isin(["NSE", "BSE", "NFO", "MCX"])]

        # Normalize text for searching
        df["symbol"] = df["symbol"].str.upper()
        df["name"] = df["name"].str.upper()

        df["search_text"] = df["symbol"] + " " + df["name"]

        INSTRUMENTS_DF = df
        print(f"✔ Loaded {len(df)} instruments from CSV.")
    except Exception as e:
        print(f"Error loading instruments: {e}")
        INSTRUMENTS_DF = pd.DataFrame()


# Load on startup
if INSTRUMENTS_DF is None:
    load_instruments()


def search_symbols(query: str, market_type: str = "EQUITY", limit: int = 10) -> List[Dict]:
    """
    Fast search for autocomplete.
    """
    q = query.upper().strip()
    if not q: return []

    # --- CRYPTO HANDLING (CoinDCX / WazirX Pairs) ---
    if market_type == "CRYPTO":
        # Expanded List of Top Liquid Pairs
        top_crypto = [
            "BTCINR", "ETHINR", "USDTINR", "XRPINR", "SOLINR", "DOGEINR",
            "ADAINR", "MATICINR", "TRXINR", "LTCINR", "SHIBINR", "BNBINR",
            "DOTINR", "AVAXINR", "LINKINR", "UNIINR", "ATOMINR", "XMRINR",
            "BCHINR", "FILINR", "NEARINR", "ALGOINR", "MANAINR", "SANDINR",
            "EOSINR", "AAVEINR", "AXSINR", "GRTINR", "FTMINR", "GALAINR",
            "CHZINR", "ENJINR", "BATINR", "ZILINR", "ETCINR", "VETINR"
        ]

        matches = [c for c in top_crypto if q in c]
        # Sort matches: items starting with query first
        matches.sort(key=lambda x: 0 if x.startswith(q) else 1)

        return [
            {"symbol": m, "name": f"Crypto {m}", "token": m, "exchange": "COINDCX", "market": "CRYPTO"}
            for m in matches[:limit]
        ]

    # --- EQUITY / F&O HANDLING (SmartAPI) ---
    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
        return []

    # 1. Filter by Query
    mask = INSTRUMENTS_DF["search_text"].str.contains(q, na=False)
    matches = INSTRUMENTS_DF[mask].copy()

    # 2. Filter by Market Type (With Crash Fixes)
    if market_type == "EQUITY":
        matches = matches[
            (matches["exch_seg"].isin(["NSE", "BSE"])) &
            (~matches["instrumenttype"].str.contains("FUT|OPT", regex=True, na=False)) &  # Added na=False
            (~matches["symbol"].str.contains("NIFTY|BANKNIFTY", regex=True, na=False))  # Added na=False
            ]

    elif market_type == "FUTURES":
        matches = matches[
            (matches["exch_seg"] == "NFO") &
            (matches["instrumenttype"].str.contains("FUT", na=False))
            ]

    elif market_type == "OPTIONS":
        matches = matches[
            (matches["exch_seg"] == "NFO") &
            (matches["instrumenttype"].str.contains("OPT", na=False))
            ]

    # 3. Format Output
    results = []
    # Sort: Exact symbol match first, then name match
    matches["is_exact"] = matches["symbol"] == q
    matches = matches.sort_values(["is_exact", "symbol"], ascending=[False, True])

    for _, row in matches.head(limit).iterrows():
        name = row["name"] if row["name"] else row["symbol"]
        expiry = row["expiry"] if row["expiry"] else ""

        # Nice Label
        if market_type in ["FUTURES", "OPTIONS"]:
            label = f"{row['symbol']} ({expiry})"
        else:
            label = f"{row['symbol']} - {name}"

        results.append({
            "symbol": row["symbol"],
            "name": label,
            "token": row["token"],
            "exchange": row["exch_seg"],
            "expiry": expiry,
            "market": market_type
        })

    return results


def find_symbol_token(symbol: str, exchange: str = "NSE"):
    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty: return None
    match = INSTRUMENTS_DF[
        (INSTRUMENTS_DF["symbol"] == symbol) &
        (INSTRUMENTS_DF["exch_seg"] == exchange)
        ]
    if not match.empty:
        return match.iloc[0]["token"]
    return None