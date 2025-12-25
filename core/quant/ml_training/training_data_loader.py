# core/quant/ml_training/training_data_loader.py

import requests
import pandas as pd
import time


class TrainingDataLoader:
    BASE_FUT = "https://fapi.binance.com/fapi/v1"

    @staticmethod
    def fetch_deep_history(symbol, interval="1h", limit=50000):
        """
        Fetches massive historical data for ML training.
        Bypasses Django cache to ensure fresh, deep data.
        """
        print(f"🚀 DEEP FETCH: Downloading {limit} candles for {symbol}...")

        klines = []
        start_ts = None
        batch_size = 1500  # Binance Max

        try:
            while len(klines) < limit:
                params = {"symbol": symbol, "interval": interval, "limit": batch_size}
                if start_ts:
                    params["endTime"] = start_ts

                # Retry logic for stability
                for _ in range(3):
                    try:
                        resp = requests.get(f"{TrainingDataLoader.BASE_FUT}/klines", params=params, timeout=10)
                        if resp.status_code == 200: break
                        time.sleep(1)
                    except:
                        time.sleep(1)

                if resp.status_code != 200:
                    print(f"   ⚠️ API Error: {resp.status_code}")
                    break

                batch = resp.json()
                if not isinstance(batch, list) or len(batch) == 0: break

                klines = batch + klines
                start_ts = batch[0][0] - 1

                if len(klines) % 5000 < batch_size:
                    print(f"   ...fetched {len(klines)} candles")

                if len(batch) < batch_size: break

            if not klines: return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base",
                "taker_quote", "ignore"
            ])

            cols = ["open", "high", "low", "close", "volume", "taker_base"]
            df[cols] = df[cols].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            print(f"❌ Critical Fetch Error: {e}")
            return pd.DataFrame()