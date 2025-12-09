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
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols, dtype={"token": str})

        # Clean NaN values to prevent JSON serialization crash
        df = df.fillna("")

        # Filter only valid segments
        df = df[df["exch_seg"].isin(["NSE", "BSE", "NFO", "MCX"])]

        # Normalize text for searching
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["name"] = df["name"].astype(str).str.upper()

        df["search_text"] = (df["symbol"] + " " + df["name"]).str.upper()

        INSTRUMENTS_DF = df
        print(f"✔ Loaded {len(df)} instruments from CSV.")

    except Exception as e:
        print(f"Error loading instruments.csv: {e}")
        INSTRUMENTS_DF = pd.DataFrame()


def search_symbols(query: str, market_type: str = "EQUITY", limit: int = 10) -> List[Dict]:
    """
    Fully cleaned autosuggestion.
    Prevents:
    - Showing indices in equity search
    - Nan crashes
    - Wrong matches
    """
    global INSTRUMENTS_DF

    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
        load_instruments()
        if INSTRUMENTS_DF.empty:
            return []

    q = query.strip().upper()
    if not q:
        return []

    df = INSTRUMENTS_DF

    # Base filter
    matches = df[df["search_text"].str.contains(q, na=False)]

    # MARKET FILTERS
    if market_type == "EQUITY":
        matches = matches[
            (matches["exch_seg"].isin(["NSE", "BSE"])) &
            (~matches["instrumenttype"].str.contains("FUT|OPT", regex=True)) &
            (~matches["symbol"].str.contains("NIFTY|BANKNIFTY|MIDCAP|SENSEX"))
        ]

    elif market_type == "FUTURES":
        matches = matches[
            (matches["exch_seg"] == "NFO") &
            (matches["instrumenttype"].str.contains("FUT"))
        ]

    elif market_type == "OPTIONS":
        matches = matches[
            (matches["exch_seg"] == "NFO") &
            (matches["instrumenttype"].str.contains("OPT"))
        ]

    elif market_type == "CRYPTO":
        # WazirX custom list
        crypto_pairs = ["BTCINR", "ETHINR", "USDTINR", "WRXINR", "MATICINR"]
        return [
            {"symbol": p, "name": p, "token": p, "exchange": "WAZIRX", "market": "CRYPTO"}
            for p in crypto_pairs if q in p
        ][:limit]

    # Format + clean output
    final = []
    for _, row in matches.head(limit).iterrows():

        clean_name = row["name"].title() if row["name"] else row["symbol"]
        expiry = f" ({row['expiry']})" if row["expiry"] else ""

        final.append({
            "symbol": row["symbol"],
            "name": clean_name + expiry,
            "token": row["token"].strip(),
            "exchange": row["exch_seg"],
            "market": market_type,
        })

    return final


# ============================================================
#  FIND SYMBOL TOKEN (MISSING EARLIER — RESTORED NOW)
# ============================================================

def find_symbol_token(symbol: str) -> str:
    """
    Returns SmartAPI instrument token for:
    - Equity (NSE/BSE)
    - Index (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
    - Futures (NIFTY25JANFUT)
    - Options (NIFTY25JAN22500CE)
    """

    global INSTRUMENTS_DF

    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
        load_instruments()
        if INSTRUMENTS_DF.empty:
            return None

    s = symbol.strip().upper()

    df = INSTRUMENTS_DF

    # ----------------------------------------------------------
    #  If exact symbol match exists → BEST case
    # ----------------------------------------------------------
    exact = df[df["symbol"] == s]
    if not exact.empty:
        return exact.iloc[0]["token"]

    # ----------------------------------------------------------
    #  Options contract detection (NIFTY25JAN22500CE)
    # ----------------------------------------------------------
    if len(s) > 10 and ("CE" in s or "PE" in s):
        opt = df[
            (df["symbol"].str.contains(s[:10])) &
            (df["instrumenttype"].str.contains("OPT"))
        ]
        if not opt.empty:
            return opt.iloc[0]["token"]

    # ----------------------------------------------------------
    #  Futures detection (NIFTY25JANFUT)
    # ----------------------------------------------------------
    if s.endswith("FUT"):
        fut = df[
            (df["symbol"].str.contains(s.replace("FUT", ""))) &
            (df["instrumenttype"].str.contains("FUT"))
        ]
        if not fut.empty:
            return fut.iloc[0]["token"]

    # ----------------------------------------------------------
    #  Index fallback (FINNIFTY, BANKNIFTY, MIDCPNIFTY)
    # ----------------------------------------------------------
    idx_map = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCPNIFTY": "MIDCPNIFTY",
    }

    for key, val in idx_map.items():
        if key in s:
            idx = df[df["symbol"] == val]
            if not idx.empty:
                return idx.iloc[0]["token"]

    # ----------------------------------------------------------
    #  Looser match for equities
    # ----------------------------------------------------------
    loose = df[df["symbol"].str.contains(s)]
    if not loose.empty:
        return loose.iloc[0]["token"]

    return None
