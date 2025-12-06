# core/services/search_service.py

import pandas as pd
import os
from django.conf import settings
from typing import List, Dict

INSTRUMENTS_DF = None


def load_instruments():
    global INSTRUMENTS_DF
    csv_path = os.path.join(settings.BASE_DIR, 'instruments.csv')

    if os.path.exists(csv_path):
        try:
            # Force token to string to preserve leading zeros if any
            cols = ['token', 'symbol', 'name', 'exch_seg', 'instrumenttype', 'expiry']
            df = pd.read_csv(csv_path, usecols=lambda c: c in cols, dtype={'token': str})

            # Filter valid segments immediately
            df = df[df['exch_seg'].isin(['NSE', 'NFO', 'BSE', 'MCX'])]

            # Create a robust search column
            df['search_text'] = df['symbol'].astype(str) + " " + df['name'].astype(str)
            df['search_text'] = df['search_text'].str.upper()

            INSTRUMENTS_DF = df
            print(f"Loaded {len(df)} instruments from CSV.")
        except Exception as e:
            print(f"Error loading instruments.csv: {e}")
            INSTRUMENTS_DF = pd.DataFrame()
    else:
        print(f"instruments.csv not found at {csv_path}")
        INSTRUMENTS_DF = pd.DataFrame()


def search_symbols(query: str, market_type: str = "EQUITY", limit: int = 10) -> List[Dict]:
    global INSTRUMENTS_DF
    if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
        load_instruments()
        if INSTRUMENTS_DF is None or INSTRUMENTS_DF.empty:
            return []

    q = query.upper().strip()
    if not q: return []

    results = []

    # --- CRYPTO (WazirX) ---
    if market_type == "CRYPTO":
        pairs = ["BTCINR", "ETHINR", "USDTINR", "WRXINR", "MATICINR", "DOGEINR", "SHIBINR", "SOLINR", "XRPINR"]
        for p in pairs:
            if q in p:
                results.append({"symbol": p, "name": p, "token": p, "exchange": "WAZIRX", "market": "CRYPTO"})
        return results[:limit]

    # --- ANGEL ONE (SmartAPI) ---
    # 1. Base Filter
    mask = INSTRUMENTS_DF['search_text'].str.contains(q, na=False)
    matches = INSTRUMENTS_DF[mask]

    # 2. Strict Segment Filtering
    if market_type == "EQUITY":
        # NSE/BSE Equity ONLY. Explicitly exclude Derivatives.
        # Note: Indices like Nifty 50 often appear in NSE but are not tradable equity.
        # We generally want stocks (EQ) or Indices if intended.
        matches = matches[
            matches['exch_seg'].isin(['NSE', 'BSE']) &
            ~matches['instrumenttype'].astype(str).str.contains('FUT|OPT', regex=True)
            ]

    elif market_type == "FUTURES":
        # NFO Futures ONLY.
        matches = matches[
            (matches['exch_seg'] == 'NFO') &
            (matches['instrumenttype'].astype(str).str.contains('FUT', case=False))
            ]
        # Sort by expiry (nearest first) to show active contracts
        if 'expiry' in matches.columns:
            matches = matches.sort_values('expiry')

    elif market_type == "OPTIONS":
        # NFO Options ONLY.
        matches = matches[
            (matches['exch_seg'] == 'NFO') &
            (matches['instrumenttype'].astype(str).str.contains('OPT', case=False))
            ]
        # Sort by expiry to prioritize near-term contracts
        if 'expiry' in matches.columns:
            matches = matches.sort_values('expiry')

    # 3. Format Results
    final_res = []
    for _, row in matches.head(limit).iterrows():
        name_str = str(row['name']) if pd.notna(row['name']) else row['symbol']
        expiry_str = f" ({row['expiry']})" if pd.notna(row['expiry']) and row['expiry'] != "" else ""

        final_res.append({
            "symbol": row['symbol'],
            "name": f"{name_str}{expiry_str}",
            "token": str(row['token']).strip(),  # CRITICAL: Ensure string and no whitespace
            "exchange": row['exch_seg'],
            "market": market_type
        })

    return final_res