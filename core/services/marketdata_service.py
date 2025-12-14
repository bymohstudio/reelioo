import requests
import pandas as pd
import time
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
        """
        Loads all valid symbols from Binance (Spot + Futures).
        Cached for 10 minutes to save API weight.
        """
        cache_key = "exchange_info_v1"
        cached = cache.get(cache_key)
        if cached:
            return cached

        try:
            # Fetch concurrently (conceptually)
            spot = requests.get(f"{MarketService.BASE_SPOT}/exchangeInfo", timeout=5).json()
            fut = requests.get(f"{MarketService.BASE_FUT}/exchangeInfo", timeout=5).json()

            results = []

            # 1. SPOT USDT
            if "symbols" in spot:
                for s in spot["symbols"]:
                    if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                        results.append({
                            "symbol": s["symbol"],
                            "name": s["baseAsset"],
                            "type": "SPOT"
                        })

            # 2. FUTURES USDT (Perps + Delivery)
            if "symbols" in fut:
                for s in fut["symbols"]:
                    if s["contractType"] in ["PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"]:
                        if s["quoteAsset"] == "USDT":
                            results.append({
                                "symbol": s["symbol"],
                                "name": s["baseAsset"],
                                "type": "PERP" if s["contractType"] == "PERPETUAL" else "FUTURES"
                            })

            # Save to Cache
            cache.set(cache_key, results, timeout=600)
            return results

        except Exception as e:
            log.error(f"Exchange Info Error: {e}")
            return []

    # -------------------------
    #  2. SMART SEARCH
    # -------------------------
    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query: return []

        # Use Cached List
        all_symbols = MarketService._load_exchange_info()

        # Fast Filter
        matches = [
            s for s in all_symbols
            if query in s["symbol"] or query in s["name"]
        ]

        # Sort: Exact match first, then starts with, then contains
        matches.sort(key=lambda x: (
            x["symbol"] != query,
            not x["symbol"].startswith(query)
        ))

        return matches[:10]

    # -------------------------
    #  3. HISTORICAL DATA (The Engine)
    # -------------------------
    @staticmethod
    def get_historical_data(symbol_input, market_type="SPOT", trade_style="SWING"):
        """
        Robust fetcher handling Spot, Perps, and Delivery Futures.
        """
        # A. Resolve Symbol
        # If input is a dict (from search), use it directly.
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
            market_type = symbol_input.get("type", market_type)  # Auto-switch market type
        else:
            # Manual Input Cleanup
            # FIX: Do NOT remove underscores (_) to support Quarterlies like BTCUSDT_250627
            raw = str(symbol_input).upper().strip().replace("/", "").replace("-", "")

            # Auto-append USDT if missing (and not a known pair)
            if "USDT" not in raw and "BUSD" not in raw:
                symbol = f"{raw}USDT"
            else:
                symbol = raw

        # B. Map Timeframe
        interval_map = {
            "SCALP": "15m",
            "INTRADAY": "1h",
            "SWING": "4h",
            "POSITION": "1d"
        }
        interval = interval_map.get(trade_style, "1h")
        limit = 1000  # Standard depth for ML

        # C. Check Cache (SaaS Speed Layer)
        cache_key = f"kline:{symbol}:{interval}:{market_type}"
        cached_df = cache.get(cache_key)
        if cached_df is not None:
            return cached_df

        # D. Determine API Endpoint (Critical Fix)
        # Handle "PERP", "FUTURES", and "SWAP" as Futures API
        if market_type in ["PERP", "FUTURES", "SWAP"]:
            base_url = f"{MarketService.BASE_FUT}/klines"
        else:
            base_url = f"{MarketService.BASE_SPOT}/klines"

        # E. Fetch from Binance
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = requests.get(base_url, params=params, timeout=5)
            data = response.json()

            # Handle API Errors (e.g., Symbol Invalid)
            if isinstance(data, dict) and "code" in data:
                log.warning(f"Binance API Error: {data}")
                return pd.DataFrame()

            # F. Parse to DataFrame
            # Binance Klines: [Time, Open, High, Low, Close, Vol, ...]
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base", "taker_quote", "ignore"
            ])

            # Numeric Conversion
            cols = ["open", "high", "low", "close", "volume"]
            for c in cols:
                df[c] = df[c].astype(float)

            # DateTime Index
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            final_df = df[cols]

            # G. Save to Cache (5 Minutes)
            # This protects your API limits when 1000 users scan BTC at once
            if not final_df.empty:
                cache.set(cache_key, final_df, timeout=300)

            return final_df

        except Exception as e:
            log.error(f"Market Data Crash: {e}")
            return pd.DataFrame()