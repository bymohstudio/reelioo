# core/services/marketdata_service.py

import datetime
import logging
import pandas as pd
import yfinance as yf
import os

try:
    from SmartApi import SmartConnect
except ImportError:
    SmartConnect = None

log = logging.getLogger(__name__)


class SmartApiClientWrapper:
    def __init__(self):
        self.hist_client = None
        self.live_client = None
        self._load_env()

    def _load_env(self):
        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.mpin = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP_KEY")
        self.hist_key = os.environ.get("SMART_HIST_API_KEY") or os.environ.get("SMART_API_KEY")
        self.live_key = os.environ.get("SMART_LIVE_API_KEY") or os.environ.get("SMART_API_KEY")

    def _connect(self, key):
        if not SmartConnect or not key: return None
        try:
            c = SmartConnect(api_key=key)
            if self.mpin and self.totp_key:
                import pyotp
                t = pyotp.TOTP(self.totp_key).now()
                d = c.generateSession(self.client_id, self.mpin, t)
                if d and d.get('status'): return c
        except:
            pass
        return None

    def get_hist(self):
        if not self.hist_client: self.hist_client = self._connect(self.hist_key)
        return self.hist_client

    def get_live(self):
        if not self.live_client: self.live_client = self._connect(self.live_key)
        return self.live_client


smartapi_wrapper = SmartApiClientWrapper()


def smartapi_get_ohlc(token, exchange, timeframe, days):
    client = smartapi_wrapper.get_hist()
    if not client: return None
    try:
        # Interval Mapping
        intv = "ONE_DAY"
        if timeframe == "5m":
            intv = "FIVE_MINUTE"
        elif timeframe == "15m":
            intv = "FIFTEEN_MINUTE"

        to_d = datetime.datetime.now()
        from_d = to_d - datetime.timedelta(days=days)

        p = {
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": intv,
            "fromdate": from_d.strftime("%Y-%m-%d %H:%M"),
            "todate": to_d.strftime("%Y-%m-%d %H:%M")
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


def yahoo_get_ohlc(symbol, timeframe, days, market_type="EQUITY"):
    try:
        # Yahoo Limit Clamp
        safe_days = days
        if timeframe in ["1m", "5m", "15m"] and days > 59: safe_days = 59

        tick = symbol
        if market_type == "EQUITY" and not tick.endswith(".NS"):
            tick += ".NS"
        elif market_type in ["FUTURES", "OPTIONS"]:
            if "NIFTY" in tick: tick = "^NSEI"
            if "BANK" in tick: tick = "^NSEBANK"
        elif market_type == "CRYPTO":
            if "INR" in tick:
                tick = tick
            else:
                tick += "-USD"

        df = yf.download(tick, period=f"{safe_days}d", interval=("1d" if timeframe == "1d" else timeframe),
                         progress=False, auto_adjust=True)
        if df.empty: return None

        # Clean Columns
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


class MarketService:
    @staticmethod
    def get_historical_data(symbol_info, market_type, trade_style):
        s = symbol_info.get("symbol")
        t = symbol_info.get("token")

        # Backtest needs history
        days = 365
        tf = "1d"
        if trade_style == "INTRADAY":
            tf = "5m";
            days = 5  # Live limit
        elif trade_style == "SWING":
            tf = "15m";
            days = 59

        # Try SmartAPI
        if t and market_type in ["EQUITY", "FUTURES"]:
            df = smartapi_get_ohlc(t, "NSE" if market_type == "EQUITY" else "NFO", tf, days)
            if df is not None: return df

        # Fallback Yahoo
        return yahoo_get_ohlc(s, tf, days, market_type)