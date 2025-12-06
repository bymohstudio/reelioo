# core/services/search_service.py

import pandas as pd
import os
from django.conf import settings
from typing import List, Dict

# Global Cache to avoid reloading CSV on every request
INSTRUMENTS_DF = None

def load_instruments():
    """
    Loads Angel One instruments.csv into memory.
    Columns needed: token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg
    """
    global INSTRUMENTS_DF
    # Assumes instruments.csv is in the project root (BASE_DIR)
    csv_path = os.path.join(settings.BASE_DIR, 'instruments.csv')
    
    if os.path.exists(csv_path):
        try:
            # Read only essential columns for performance
            # Dtype specified for token to prevent leading zero loss
            cols = ['token', 'symbol', 'name', 'exch_seg', 'instrumenttype', 'expiry']
            df = pd.read_csv(csv_path, usecols=lambda c: c in cols, dtype={'token': str})
            
            # Filter for NSE/NFO/BSE/MCX only (excluding others if needed)
            df = df[df['exch_seg'].isin(['NSE', 'NFO', 'BSE', 'MCX'])]
            
            # Create a combined search column for speed
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
        # Since we don't have a WazirX CSV, we use a static top list + query match
        # or you could fetch WazirX /api/v2/tickers and cache it.
        # Here is a robust static list for Indian context
        pairs = [
            "BTCINR", "ETHINR", "USDTINR", "WRXINR", "MATICINR", "DOGEINR", 
            "SHIBINR", "SOLINR", "XRPINR", "TRXINR", "GALAINR", "SANDINR"
        ]
        for p in pairs:
            if q in p:
                results.append({
                    "symbol": p, 
                    "name": p, 
                    "token": p, # WazirX API uses symbol as token
                    "exchange": "WAZIRX", 
                    "market": "CRYPTO"
                })
        return results[:limit]

    # --- ANGEL ONE (SmartAPI) ---
    
    # 1. Base Filter by Query
    mask = INSTRUMENTS_DF['search_text'].str.contains(q, na=False)
    matches = INSTRUMENTS_DF[mask]
    
    # 2. Refine by Market Type
    if market_type == "EQUITY":
        # NSE or BSE, exclude Futures/Options
        matches = matches[
            matches['exch_seg'].isin(['NSE', 'BSE']) & 
            ~matches['instrumenttype'].astype(str).str.contains('FUT|OPT', regex=True)
        ]
        # Prioritize NSE
        matches = matches.sort_values(by='exch_seg', ascending=False) # NSE usually comes after BSE alphabetically? No. 
        # Custom sort: NSE first
        matches['sort_rank'] = matches['exch_seg'].apply(lambda x: 0 if x == 'NSE' else 1)
        matches = matches.sort_values('sort_rank')

    elif market_type == "FUTURES":
        # NFO segment, instrument contains FUT
        matches = matches[
            (matches['exch_seg'] == 'NFO') & 
            (matches['instrumenttype'].astype(str).str.contains('FUT', case=False))
        ]
        # Sort by expiry (nearest first) - simplified for now
        matches = matches.sort_values('expiry')

    elif market_type == "OPTIONS":
        # NFO segment, instrument contains OPT
        matches = matches[
            (matches['exch_seg'] == 'NFO') & 
            (matches['instrumenttype'].astype(str).str.contains('OPT', case=False))
        ]
        # Options have huge volume of data, limit strictness
        matches = matches.sort_values('expiry')

    # 3. Format Results
    final_res = []
    for _, row in matches.head(limit).iterrows():
        name_str = str(row['name']) if pd.notna(row['name']) else row['symbol']
        expiry_str = f" ({row['expiry']})" if pd.notna(row['expiry']) and row['expiry'] != "" else ""
        
        final_res.append({
            "symbol": row['symbol'],
            "name": f"{name_str}{expiry_str}",
            "token": str(row['token']),
            "exchange": row['exch_seg'],
            "market": market_type
        })
        
    return final_res