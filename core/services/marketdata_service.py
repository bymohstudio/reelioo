import requests
import pandas as pd
import logging
from django.core.cache import cache

log = logging.getLogger(__name__)


class MarketService:
    BASE_SPOT = "https://api.binance.com/api/v3"
    BASE_FUT = "https://fapi.binance.com/fapi/v1"

    # -------------------------
    #  1. EXCHANGE INFO (Cached)
    # -------------------------
    @staticmethod
    def _load_exchange_info():
        cache_key = "exchange_info_v1"
        cached = cache.get(cache_key)
        if cached: return cached

        try:
            # HEADERS ARE CRITICAL TO PREVENT 403 ERRORS
            headers = {"User-Agent": "Mozilla/5.0"}
            spot = requests.get(f"{MarketService.BASE_SPOT}/exchangeInfo", headers=headers, timeout=5).json()
            fut = requests.get(f"{MarketService.BASE_FUT}/exchangeInfo", headers=headers, timeout=5).json()
            results = []

            # SPOT
            if "symbols" in spot:
                for s in spot["symbols"]:
                    if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                        results.append({"symbol": s["symbol"], "name": s["baseAsset"], "type": "SPOT"})

            # FUTURES
            if "symbols" in fut:
                for s in fut["symbols"]:
                    if s["contractType"] in ["PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"]:
                        if s["quoteAsset"] == "USDT":
                            results.append({
                                "symbol": s["symbol"],
                                "name": s["baseAsset"],
                                "type": "PERP" if s["contractType"] == "PERPETUAL" else "FUTURES"
                            })

            cache.set(cache_key, results, timeout=600)
            return results
        except Exception as e:
            log.error(f"Exchange Info Error: {e}")
            return []

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
        # 1. Resolve Symbol
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
            market_type = symbol_input.get("type", market_type)
        else:
            raw = str(symbol_input).upper().strip().replace("/", "").replace("-", "")
            symbol = f"{raw}USDT" if "USDT" not in raw else raw

        # 2. Map Timeframe
        interval_map = {
            "SCALP": "15m",
            "INTRADAY": "1h",
            "SWING": "4h",
            "POSITION": "1d"
        }
        interval = interval_map.get(trade_style, "1h")
        limit = 1000

        cache_key = f"kline_v3:{symbol}:{interval}:{market_type}"
        cached_df = cache.get(cache_key)
        if cached_df is not None: return cached_df

        # 3. Fetch
        base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT

        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            # HEADERS REQUIRED FOR BINANCE (Bypasses 403 Forbidden)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

            print(f"🔄 Fetching Data: {symbol} [{interval}] from {base_url}...")

            resp = requests.get(f"{base_url}/klines", params=params, headers=headers, timeout=10)

            if resp.status_code != 200:
                print(f"❌ Binance Error {resp.status_code}: {resp.text}")
                log.error(f"Binance API Error {resp.status_code}: {resp.text}")
                return pd.DataFrame()

            data = resp.json()

            if isinstance(data, dict) and "code" in data:
                print(f"❌ API Error Code: {data}")
                return pd.DataFrame()

            if not data or len(data) == 0:
                print("❌ No data returned from API")
                return pd.DataFrame()

            # --- CRITICAL FIX: INCLUDE TAKER BUY VOLUME ---
            # Index 9 is 'Taker buy base asset volume' - Required for Order Flow Features
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base", "taker_quote", "ignore"
            ])

            # Keep taker_base so Feature Engineering can calculate CVD
            cols_to_keep = ["open", "high", "low", "close", "volume", "taker_base"]

            df[cols_to_keep] = df[cols_to_keep].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # RENAME taker_base -> taker_buy_base to match feature_engineering.py
            df = df.rename(columns={"taker_base": "taker_buy_base"})

            final_df = df[["open", "high", "low", "close", "volume", "taker_buy_base"]]

            if not final_df.empty:
                print(f"✅ Data Success: {len(final_df)} candles")
                cache.set(cache_key, final_df, timeout=300)

            return final_df

        except Exception as e:
            print(f"❌ Exception in MarketService: {e}")
            log.error(f"Data Fetch Error: {e}")
            return pd.DataFrame()