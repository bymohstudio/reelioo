# core/quant/ml_training/fetch_data.py

import os
import pandas as pd
import datetime
import time
import logging
from SmartApi import SmartConnect
import pyotp
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TrainData")


class SmartTrainSession:
    _instance = None
    _instruments = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SmartTrainSession, cls).__new__(cls)
            cls._instance.client = None
            cls._instance._login()
            cls._instance._load_instruments()
        return cls._instance

    def _login(self):
        try:
            api_key = os.environ.get("SMART_HIST_API_KEY") or os.environ.get("SMART_API_KEY")
            client_id = os.environ.get("SMART_CLIENT_ID")
            mpin = os.environ.get("SMART_PASSWORD")
            totp_key = os.environ.get("SMART_TOTP_KEY")

            if not (api_key and client_id and mpin and totp_key):
                return

            self.client = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_key).now()
            data = self.client.generateSession(client_id, mpin, totp)

            if data and data.get("status"):
                log.info("✅ SmartAPI Connected")
            else:
                self.client = None
        except Exception as e:
            self.client = None

    def _load_instruments(self):
        csv_path = os.path.join(os.getcwd(), "instruments.csv")
        if not os.path.exists(csv_path): return
        try:
            df = pd.read_csv(csv_path, usecols=["symbol", "token", "exch_seg", "name"], dtype=str)
            self._instruments = df
        except Exception:
            pass

    def get_token(self, symbol, exchange="NSE"):
        if self._instruments is None: return None
        subset = self._instruments[self._instruments["exch_seg"] == exchange]
        match = subset[subset["symbol"] == symbol]
        if not match.empty: return match.iloc[0]["token"]
        match = subset[subset["symbol"].str.contains(symbol)]
        if not match.empty: return match.iloc[0]["token"]
        return None


class DataFetcher:

    @staticmethod
    def fetch(symbol: str, market: str, days: int = 730):
        # 1. FORCE YAHOO FOR INDICES (Most reliable for training history)
        if symbol in ["NIFTY", "BANKNIFTY"]:
            return DataFetcher.fetch_yahoo(symbol, days)

        # 2. Try SmartAPI for Stocks
        session = SmartTrainSession()
        exchange = "NSE"

        if market == "CRYPTO":
            return DataFetcher.fetch_yahoo(symbol, days)

        token = session.get_token(symbol, exchange)

        if session.client and token:
            try:
                log.info(f"⬇ SmartAPI: {symbol}...")
                to_date = datetime.datetime.now()
                from_date = to_date - datetime.timedelta(days=days)

                params = {
                    "exchange": exchange,
                    "symboltoken": token,
                    "interval": "ONE_DAY",
                    "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                    "todate": to_date.strftime("%Y-%m-%d %H:%M")
                }

                res = session.client.getCandleData(params)
                if res and res.get("data"):
                    records = []
                    for r in res["data"]:
                        records.append({
                            "timestamp": pd.to_datetime(r[0]),
                            "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
                            "volume": float(r[5])
                        })
                    df = pd.DataFrame(records)
                    return df[["open", "high", "low", "close", "volume"]]
            except Exception:
                pass

        # 3. Fallback
        return DataFetcher.fetch_yahoo(symbol, days)

    @staticmethod
    def fetch_yahoo(symbol, days):
        import yfinance as yf
        try:
            ticker = f"{symbol}.NS"
            if symbol == "NIFTY": ticker = "^NSEI"
            if symbol == "BANKNIFTY": ticker = "^NSEBANK"
            if "INR" in symbol: ticker = symbol  # Crypto

            print(f"  > Yahoo Fetch: {ticker}")
            df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)

            if df.empty: return None

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
            return df
        except Exception:
            return None