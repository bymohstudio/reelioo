# core/services/marketdata_service.py

import datetime
import logging
import pandas as pd
import yfinance as yf
import os

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
    """
    Manages TWO separate connections:
    1. Historical Data Client (for Backtesting/Charts)
    2. Live Feed Client (for Real-time Price updates)
    """

    def __init__(self):
        self.hist_client = None
        self.live_client = None
        self.logged_in_hist = False
        self.logged_in_live = False

        # Shared Credentials
        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.mpin = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP_KEY")

        # Separate API Keys (Fallback to generic SMART_API_KEY if specific not found)
        self.hist_api_key = os.environ.get("SMART_HIST_API_KEY") or os.environ.get("SMART_API_KEY")
        self.live_api_key = os.environ.get("SMART_LIVE_API_KEY") or os.environ.get("SMART_API_KEY")

    def _create_session(self, api_key):
        """Helper to create a fresh session"""
        if not SmartConnect or not api_key or not self.client_id:
            return None
        try:
            client = SmartConnect(api_key=api_key)
            if self.mpin and self.totp_key:
                import pyotp
                totp = pyotp.TOTP(self.totp_key).now()
                data = client.generateSession(self.client_id, self.mpin, totp)
                if data and data.get("status"):
                    return client
        except Exception as e:
            log.error(f"SmartAPI Session Failed for key {api_key[:4]}...: {e}")
        return None

    def get_hist(self):
        """Returns the Historical Data Client"""
        if not self.hist_client:
            log.info("[SmartAPI] Connecting Historical Client...")
            self.hist_client = self._create_session(self.hist_api_key)
            if self.hist_client: self.logged_in_hist = True
        return self.hist_client

    def get_live(self):
        """Returns the Live Market Feed Client"""
        if not self.live_client:
            log.info("[SmartAPI] Connecting Live Feed Client...")
            self.live_client = self._create_session(self.live_api_key)
            if self.live_client: self.logged_in_live = True
        return self.live_client

    def get(self):
        # Fallback for legacy calls (defaults to Live)
        return self.get_live()


smartapi_wrapper = SmartApiClientWrapper()

# =========================================
# 2. SMARTAPI FETCHER (Routing Logic)
# =========================================
TIMEFRAME_MAP = {
    "1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "1d": "ONE_DAY"
}


def get_smart_ltp(symbol_token, exchange):
    """
    Get Real-Time Price using Live Feed API
    """
    client = smartapi_wrapper.get_live()
    if not client: return None
    try:
        data = client.ltpData(exchange, symbol_token, symbol_token)  # SDK specific, mostly ltpData or getLTP
        if data and data.get("data"):
            return float(data["data"]["ltp"])
    except:
        pass
    return None


def smartapi_get_ohlc(symbol_token, exchange, timeframe, days):
    """
    Get Candles using Historical Data API
    """
    # Use HISTORICAL client
    client = smartapi_wrapper.get_hist()
    if not client:
        return None

    try:
        interval = TIMEFRAME_MAP.get(timeframe, "FIFTEEN_MINUTE")
        to_dt = datetime.datetime.now()
        from_dt = to_dt - datetime.timedelta(days=days)

        params = {
            "exchange": exchange,
            "symboltoken": str(symbol_token),
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M")
        }

        data = client.getCandleData(params)
        if not data or not data.get("data"):
            return None

        records = []
        for r in data["data"]:
            records.append({
                "timestamp": pd.to_datetime(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5])
            })

        df = pd.DataFrame(records).set_index("timestamp")

        # --- INSTITUTIONAL EDGE: REAL-TIME PATCH ---
        # If market is open, the historical candle might be lagging by 1-2 mins.
        # We fetch the LIVE LTP and update the last close price to be instant.
        try:
            live_price = get_smart_ltp(symbol_token, exchange)
            if live_price and not df.empty:
                # Update last close to match live market price
                df.iloc[-1, df.columns.get_loc("close")] = live_price
                # Update High/Low if live price broke them
                if live_price > df.iloc[-1]["high"]: df.iloc[-1, df.columns.get_loc("high")] = live_price
                if live_price < df.iloc[-1]["low"]: df.iloc[-1, df.columns.get_loc("low")] = live_price
        except Exception:
            pass  # Fail silently, keep historical data

        return df
    except Exception as e:
        log.error(f"SmartAPI Error: {e}")
        return None


# =========================================
# 3. YAHOO FETCHER (Backup)
# =========================================
def yahoo_get_ohlc(symbol, timeframe, days, market_type="EQUITY"):
    try:
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
        interval = tf_map.get(timeframe, "1d")

        safe_days = days
        if interval == "1m":
            safe_days = min(days, 5)
        elif interval in ["5m", "15m", "30m"]:
            safe_days = min(days, 55)

        ticker = symbol
        if market_type == "EQUITY" and not ticker.endswith(".NS"):
            ticker = f"{symbol}.NS"
        elif market_type in ["FUTURES", "OPTIONS"]:
            if "BANKNIFTY" in symbol:
                ticker = "^NSEBANK"
            elif "NIFTY" in symbol:
                ticker = "^NSEI"
            elif "FINNIFTY" in symbol:
                ticker = "NIFTY_FIN_SERVICE.NS"
            elif "MIDCP" in symbol:
                ticker = "^NSEMDCP50"
            else:
                ticker = f"{symbol}.NS"
        elif market_type == "CRYPTO":
            ticker = f"{symbol.replace('INR', '')}-INR" if "INR" in symbol else f"{symbol}-USD"

        df = yf.download(ticker, period=f"{safe_days}d", interval=interval, progress=False, auto_adjust=True)

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
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        log.error(f"Yahoo Fetch Error: {e}")
        return None


# =========================================
# 4. MAIN SERVICE
# =========================================
class MarketService:
    @staticmethod
    def get_historical_data(symbol_info, market_type, trade_style):
        symbol = symbol_info.get("symbol", "")
        token = symbol_info.get("token")

        timeframe = "1d"
        days = 365

        if trade_style == "INTRADAY":
            timeframe = "5m"
            days = 5
        elif trade_style == "SWING":
            timeframe = "15m"
            days = 59

            # 1. Try SmartAPI (Institutional Routing)
        if token and market_type in ["EQUITY", "FUTURES"]:
            df = smartapi_get_ohlc(token, "NSE" if market_type == "EQUITY" else "NFO", timeframe, days)
            if df is not None and not df.empty:
                return df

        # 2. Options Special Logic
        if market_type == "OPTIONS":
            if token:
                df = smartapi_get_ohlc(token, "NFO", timeframe, days)
                if df is not None and not df.empty: return df

        # 3. Yahoo Fallback
        return yahoo_get_ohlc(symbol, timeframe, days, market_type)