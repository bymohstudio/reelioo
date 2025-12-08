# core/services/marketdata_service.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pyotp
import re

# Try importing SmartAPI
try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None


# =========================================
# 1. YAHOO FINANCE FALLBACK (The Safety Net)
# =========================================

def fetch_yahoo_fallback(symbol: str, market_type: str, days: int = 365) -> pd.DataFrame:
    """
    Fallback fetcher if SmartAPI fails or is not configured.
    Automatically handles Options by fetching the Underlying Index.
    """

    # 1. Clean Symbol (Remove -EQ, -BE)
    clean_symbol = symbol.replace("-EQ", "").replace("-BE", "").strip().upper()

    # 2. Map to Yahoo Ticker
    ticker = clean_symbol

    if market_type == "EQUITY":
        ticker = f"{clean_symbol}.NS"

    elif market_type == "FUTURES":
        # Yahoo doesn't have continuous futures. Use Index.
        if "BANKNIFTY" in clean_symbol:
            ticker = "^NSEBANK"
        elif "NIFTY" in clean_symbol:
            ticker = "^NSEI"

    elif market_type == "OPTIONS":
        # CRITICAL FIX: Yahoo has no Options data. Use Underlying.
        print(f"[DATA] Option requested ({symbol}). Yahoo Fallback -> Fetching Underlying Index.")
        if "BANKNIFTY" in clean_symbol:
            ticker = "^NSEBANK"
        elif "NIFTY" in clean_symbol:
            ticker = "^NSEI"
        elif "FINNIFTY" in clean_symbol:
            ticker = "NIFTY_FIN_SERVICE.NS"
        else:
            # Stock Option? Try fetching the stock
            # Extract basic symbol e.g., "RELIANCE24JAN..." -> "RELIANCE"
            # Simple regex to grab the first alphabetic part
            match = re.match(r"([A-Z]+)", clean_symbol)
            if match:
                ticker = f"{match.group(1)}.NS"

    elif market_type == "CRYPTO":
        ticker = "BTC-INR"

    print(f"[DATA] Yahoo Fetching: {ticker} (Strategy: {market_type})")

    # 3. Download
    try:
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)

        if df is None or df.empty:
            print(f"[DATA] Yahoo returned empty data for {ticker}")
            return pd.DataFrame()

        # 4. Clean Data (Handle MultiIndex)
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize
        df.columns = [c.lower() for c in df.columns]
        rename_map = {}
        for c in df.columns:
            if "open" in c: rename_map[c] = "open"
            if "high" in c: rename_map[c] = "high"
            if "low" in c: rename_map[c] = "low"
            if "close" in c: rename_map[c] = "close"
            if "volume" in c: rename_map[c] = "volume"

        df = df.rename(columns=rename_map)

        # Numeric Safety
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = 0.0

        df = df.dropna(subset=["close"])

        print(f"[DATA] Yahoo Success: {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        print(f"[DATA] Yahoo Fallback Failed for {ticker}: {e}")
        return pd.DataFrame()


# =========================================
# 2. SMART API PROVIDER
# =========================================

class SmartApiProvider:
    def __init__(self):
        self.api_key = os.environ.get("SMART_API_KEY")
        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.password = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP_KEY")
        self.smart_api = None
        self.logged_in = False

    def login(self):
        # Fail fast if no creds
        if not all([self.api_key, self.client_id, self.password, self.totp_key]):
            return False

        if not SmartConnect:
            return False

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_key).now()
            data = self.smart_api.generateSession(self.client_id, self.password, totp)

            if data['status']:
                self.logged_in = True
                print("[DATA] SmartAPI Login Successful")
                return True
            else:
                print(f"[DATA] SmartAPI Login Failed: {data['message']}")
                return False
        except Exception as e:
            print(f"[DATA] SmartAPI Exception: {e}")
            return False

    def get_candle_data(self, symbol_token, exchange, interval="ONE_DAY", days=200):
        if not self.logged_in:
            if not self.login():
                return None

        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            historicParam = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }

            data = self.smart_api.getCandleData(historicParam)

            if data and data.get('status') and data.get('data'):
                candles = data['data']
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

                cols = ["open", "high", "low", "close", "volume"]
                df[cols] = df[cols].apply(pd.to_numeric)

                print(f"[DATA] SmartAPI Fetch Success: {len(df)} rows")
                return df
            else:
                return None

        except Exception as e:
            print(f"[DATA] SmartAPI Fetch Error: {e}")
            return None


# =========================================
# 3. MAIN MARKET SERVICE (With Fallback)
# =========================================

class MarketService:
    _smart_provider = SmartApiProvider()

    @classmethod
    def get_historical_data(cls, symbol_info: dict, market_type: str = "EQUITY",
                            trade_style: str = "SWING") -> pd.DataFrame:
        symbol = symbol_info.get("symbol")
        token = symbol_info.get("token")
        exchange = symbol_info.get("exchange")

        print(f"\n--- FETCHING DATA FOR {symbol} ({market_type}) ---")

        # 1. Try SmartAPI (Institutional)
        if token and exchange:
            interval = "ONE_DAY"
            if trade_style == "INTRADAY": interval = "FIFTEEN_MINUTE"

            df = cls._smart_provider.get_candle_data(token, exchange, interval)
            if df is not None and not df.empty:
                return df
            else:
                print("[DATA] SmartAPI returned no data (or not configured).")

        # 2. Fallback to Yahoo (Retail / Backup)
        print("[DATA] Switching to Yahoo Finance Fallback...")
        df_yahoo = fetch_yahoo_fallback(symbol, market_type)

        return df_yahoo