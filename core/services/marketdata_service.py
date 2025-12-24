import os

import requests
import pandas as pd
import logging
from django.core.cache import cache

from reelioo import settings

log = logging.getLogger(__name__)


class MarketService:
    BASE_SPOT = "https://api.binance.com/api/v3"
    BASE_FUT = "https://fapi.binance.com/fapi/v1"

    # -------------------------
    #  1. EXCHANGE INFO (Cached)
    # -------------------------
    @staticmethod
    def _load_exchange_info():
        # CHANGE KEY TO FORCE REFRESH
        cache_key = "exchange_info_v2_csv"
        cached = cache.get(cache_key)
        if cached: return cached

        results = []
        # Construct path
        global_path = os.path.join(settings.BASE_DIR, "global_symbols.csv")

        # DEBUG PRINT: Check where the server is actually looking
        print(f"📂 Looking for CSV at: {global_path}")

        # 1. Try Loading from local CSV
        if os.path.exists(global_path):
            try:
                df_csv = pd.read_csv(global_path)
                results = df_csv.to_dict("records")
                print(f"✅ Loaded {len(results)} symbols from CSV")
            except Exception as e:
                log.error(f"Error reading Global CSV: {e}")
        else:
            print("❌ CSV File not found at path.")

        # 2. Fallback to live API if results are still empty
        if not results:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                spot = requests.get(f"{MarketService.BASE_SPOT}/exchangeInfo", headers=headers, timeout=5).json()

                if "symbols" in spot:
                    for s in spot["symbols"]:
                        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                            results.append({
                                "symbol": s["symbol"],
                                "name": s["baseAsset"],
                                "type": "GLOBAL"
                            })
            except Exception as e:
                log.error(f"Binance API Fallback Failed: {e}")

        # 3. Final Static Safety Net
        if not results:
            results = [
                {"symbol": "BTCUSDT", "name": "BTC", "type": "GLOBAL"},
                {"symbol": "ETHUSDT", "name": "ETH", "type": "GLOBAL"},
                {"symbol": "SOLUSDT", "name": "SOL", "type": "GLOBAL"}
            ]

        # Cache the list for 1 hour to reduce disk/network overhead
        cache.set(cache_key, results, timeout=3600)
        return results

    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query: return []
        all_symbols = MarketService._load_exchange_info()
        matches = [s for s in all_symbols if query in s["symbol"] or query in s["name"]]
        return matches[:10]

    # -------------------------
    #  3. HISTORICAL DATA (FIXED)
    # -------------------------
    @staticmethod
    def get_historical_data(symbol_input, market_type="SPOT", trade_style="SWING"):
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
        else:
            raw = str(symbol_input).upper().strip().replace("/", "").replace("-", "")
            symbol = f"{raw}USDT" if "USDT" not in raw else raw

        interval_map = {
            "SCALP": "15m",
            "INTRADAY": "1h",
            "SWING": "4h",
            "POSITION": "1d"
        }
        interval = interval_map.get(trade_style, "1h")

        # ---- PATCH: fetch MULTIPLE pages (just like training loader) ----
        max_rows_needed = 5000
        klines = []
        base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT
        limit = 1500  # Binance hard limit
        start_ts = None

        print(f"🔄 Fetching multi-page data for {symbol} [{interval}]...")

        try:
            while len(klines) < max_rows_needed:
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                if start_ts:
                    params["endTime"] = start_ts

                resp = requests.get(f"{base_url}/klines", params=params, timeout=10)
                if resp.status_code != 200:
                    print(f"❌ Binance Error {resp.status_code}")
                    break

                batch = resp.json()
                if not isinstance(batch, list) or len(batch) == 0:
                    break

                klines = batch + klines  # prepend newest → oldest
                start_ts = batch[0][0] - 1

                # Safety
                if len(batch) < limit:
                    break

            if len(klines) == 0:
                return pd.DataFrame()

            print(f"✅ Multi-page Success: {len(klines)} candles")

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base",
                "taker_quote", "ignore"
            ])

            cols = ["open", "high", "low", "close", "volume", "taker_base"]
            df[cols] = df[cols].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            final_df = df[cols]

            # OPTIONAL caching
            cache.set(f"kline_v25:{symbol}:{interval}:{market_type}",
                      final_df, timeout=300)

            return final_df

        except Exception as e:
            log.error(f"Data Fetch Error: {e}")
            return pd.DataFrame()