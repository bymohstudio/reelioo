# core/services/marketdata_service.py

import datetime
import logging
import pandas as pd
import yfinance as yf
import requests
import os
import time

# Try importing SmartAPI
try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None

log = logging.getLogger(__name__)


# =========================================
# 1. DUAL-CLIENT WRAPPER (Live + Historical)
# =========================================
class SmartApiClientWrapper:
    def __init__(self):
        self.hist_client = None
        self.live_client = None

        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.mpin = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP_KEY")
        self.hist_api_key = os.environ.get("SMART_HIST_API_KEY") or os.environ.get("SMART_API_KEY")
        self.live_api_key = os.environ.get("SMART_LIVE_API_KEY") or os.environ.get("SMART_API_KEY")

    def _create_session(self, api_key):
        if not SmartConnect or not api_key or not self.client_id: return None
        try:
            client = SmartConnect(api_key=api_key)
            if self.mpin and self.totp_key:
                import pyotp
                totp = pyotp.TOTP(self.totp_key).now()
                data = client.generateSession(self.client_id, self.mpin, totp)
                if data and data.get("status"): return client
        except:
            pass
        return None

    def get_hist(self):
        if not self.hist_client: self.hist_client = self._create_session(self.hist_api_key)
        return self.hist_client

    def get_live(self):
        if not self.live_client: self.live_client = self._create_session(self.live_api_key)
        return self.live_client


smartapi_wrapper = SmartApiClientWrapper()


# =========================================
# 2. COINDCX FETCHER (Authenticated)
# =========================================
def fetch_coindcx_data(symbol, timeframe, limit=500):
    """
    Fetches Crypto candles from CoinDCX Public API.
    Uses API Key from .env if available.
    """
    try:
        # 1. Load Credentials
        api_key = os.environ.get("COINDCX_API_KEY")

        # 2. Format Symbol (BTCINR -> B-BTC_INR)
        # Handle cases like BTC-INR, BTCINR, etc.
        clean_sym = symbol.replace("-", "").replace("_", "").upper()

        # Determine Base and Quote
        if clean_sym.endswith("INR"):
            base = clean_sym[:-3]
            pair = f"B-{base}_INR"
        elif clean_sym.endswith("USDT"):
            base = clean_sym[:-4]
            pair = f"B-{base}_USDT"
        else:
            # Fallback assumption
            pair = f"B-{clean_sym}_INR"

        # 3. Map Timeframe
        # CoinDCX: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 1d, 3d, 1w, 1M
        interval = "1h"
        if timeframe == "1m":
            interval = "1m"
        elif timeframe == "5m":
            interval = "5m"
        elif timeframe == "15m":
            interval = "15m"
        elif timeframe == "30m":
            interval = "30m"
        elif timeframe == "1h":
            interval = "1h"
        elif timeframe == "1d":
            interval = "1d"

        url = "https://public.coindcx.com/market_data/candles"
        params = {
            "pair": pair,
            "interval": interval,
            "limit": limit
        }

        # 4. Authenticated Request Headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        if api_key:
            headers["X-AUTH-APIKEY"] = api_key

        resp = requests.get(url, params=params, headers=headers, timeout=5)

        if resp.status_code != 200:
            log.warning(f"CoinDCX API Error: {resp.status_code}")
            return None

        data = resp.json()
        if not data or not isinstance(data, list):
            return None

        # 5. Parse Data
        # Returns list of dicts
        df = pd.DataFrame(data)

        # Ensure correct numeric types
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols: df[c] = pd.to_numeric(df[c])

        # Parse Time
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df[cols]

    except Exception as e:
        log.error(f"CoinDCX Fetch Error for {symbol}: {e}")
        return None


# =========================================
# 3. YAHOO FETCHER (Backup)
# =========================================
def yahoo_get_ohlc(symbol, timeframe, days, market_type="EQUITY"):
    try:
        safe_days = days
        if timeframe in ["1m", "5m", "15m"] and days > 59: safe_days = 59

        # Clean Symbol
        ticker = symbol.replace("-EQ", "").replace("-BE", "").strip().upper()

        if market_type == "EQUITY":
            if not ticker.endswith(".NS"): ticker = f"{ticker}.NS"
        elif market_type in ["FUTURES", "OPTIONS"]:
            if "NIFTY" in ticker: ticker = "^NSEI"
            if "BANK" in ticker: ticker = "^NSEBANK"
        elif market_type == "CRYPTO":
            # Map common pairs to Yahoo format
            if "INR" in ticker and "-" not in ticker:
                ticker = f"{ticker.replace('INR', '')}-INR"
            elif "USDT" in ticker:
                ticker = f"{ticker.replace('USDT', '')}-USD"

        df = yf.download(ticker, period=f"{safe_days}d", interval=("1d" if timeframe == "1d" else timeframe),
                         progress=False, auto_adjust=True)
        if df is None or df.empty: return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        rename = {}
        for c in df.columns:
            if "open" in c:
                rename[c] = "open"
            elif "high" in c:
                rename[c] = "high"
            elif "low" in c:
                rename[c] = "low"
            elif "close" in c:
                rename[c] = "close"
            elif "vol" in c:
                rename[c] = "volume"

        df = df.rename(columns=rename)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except:
        return None


def smartapi_get_ohlc(token, exchange, timeframe, days):
    client = smartapi_wrapper.get_hist()
    if not client: return None
    try:
        interval = "ONE_DAY"
        if timeframe == "5m":
            interval = "FIVE_MINUTE"
        elif timeframe == "15m":
            interval = "FIFTEEN_MINUTE"

        to_dt = datetime.datetime.now()
        from_dt = to_dt - datetime.timedelta(days=days)

        p = {
            "exchange": exchange, "symboltoken": str(token), "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"), "todate": to_dt.strftime("%Y-%m-%d %H:%M")
        }
        d = client.getCandleData(p)
        if not d or not d.get("data"): return None

        recs = []
        for r in d["data"]:
            recs.append({
                "timestamp": pd.to_datetime(r[0]),
                "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
                "volume": float(r[5])
            })
        return pd.DataFrame(recs).set_index("timestamp")
    except:
        return None


# =========================================
# 4. MAIN ROUTER
# =========================================
class MarketService:
    @staticmethod
    def get_historical_data(symbol_info, market_type, trade_style):
        symbol = symbol_info.get("symbol", "")
        token = symbol_info.get("token")

        timeframe = "1d"
        days = 365
        if trade_style == "INTRADAY":
            timeframe = "5m"; days = 5
        elif trade_style == "SWING":
            timeframe = "15m"; days = 59

        # 1. CRYPTO ROUTING (CoinDCX -> Yahoo)
        if market_type == "CRYPTO":
            df = fetch_coindcx_data(symbol, timeframe)
            if df is not None and not df.empty: return df
            return yahoo_get_ohlc(symbol, timeframe, days, market_type)

        # 2. EQUITY/FUTURES ROUTING (SmartAPI -> Yahoo)
        if token and market_type in ["EQUITY", "FUTURES"]:
            df = smartapi_get_ohlc(token, "NSE" if market_type == "EQUITY" else "NFO", timeframe, days)
            if df is not None: return df

        # 3. OPTIONS ROUTING
        if market_type == "OPTIONS":
            if token:
                df = smartapi_get_ohlc(token, "NFO", timeframe, days)
                if df is not None: return df

        # 4. Universal Fallback
        return yahoo_get_ohlc(symbol, timeframe, days, market_type)