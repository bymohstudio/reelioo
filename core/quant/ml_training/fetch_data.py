# core/quant/ml_training/fetch_data.py

import requests
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TrainData")


class DataFetcher:
    """
    Fetches historical training data from Binance.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"

    @classmethod
    def fetch(cls, symbol: str, days: int = 365, interval: str = "1h") -> pd.DataFrame:
        """
        Fetches 'days' worth of data.
        Binance limit is 1000 candles per call, so we might need pagination if days > 40.
        For simplicity in this MVP, we fetch the max allowed (1000 candles ~ 41 days of hourly data).
        """
        pair = symbol.upper().replace("-", "").replace("_", "")
        if not pair.endswith("USDT") and not pair.endswith("BTC"):
            pair = f"{pair}USDT"

        log.info(f"[FETCH] Getting max history for {pair} ({interval})...")

        try:
            params = {"symbol": pair, "interval": interval, "limit": 1000}
            resp = requests.get(cls.BASE_URL, params=params, timeout=10)

            if resp.status_code != 200:
                log.error(f"Binance Error: {resp.text}")
                return pd.DataFrame()

            data = resp.json()
            # Columns: Open Time, Open, High, Low, Close, Volume, ...
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base", "taker_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            cols = ["open", "high", "low", "close", "volume"]
            df[cols] = df[cols].apply(pd.to_numeric)

            df.set_index("timestamp", inplace=True)
            df = df.sort_index()

            log.info(f"[FETCH] Success: {len(df)} rows for {pair}")
            return df

        except Exception as e:
            log.error(f"Fetch failed: {e}")
            return pd.DataFrame()