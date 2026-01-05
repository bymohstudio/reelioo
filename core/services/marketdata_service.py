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
        cache_key = "exchange_info_v2_csv"
        cached = cache.get(cache_key)
        if cached: return cached

        results = []
        global_path = os.path.join(settings.BASE_DIR, "global_symbols.csv")

        # 1. Try Loading from local CSV
        if os.path.exists(global_path):
            try:
                df_csv = pd.read_csv(global_path)
                results = df_csv.to_dict("records")
            except Exception as e:
                log.error(f"Error reading Global CSV: {e}")

        # 2. Fallback to live API (Futures First)
        if not results:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                fapi = requests.get(f"{MarketService.BASE_FUT}/exchangeInfo", headers=headers, timeout=5).json()
                if "symbols" in fapi:
                    for s in fapi["symbols"]:
                        if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                            results.append({"symbol": s["symbol"], "name": s["baseAsset"], "type": "PERP"})
            except Exception as e:
                log.error(f"Binance API Fallback Failed: {e}")

        # 3. Static Safety Net
        if not results:
            results = [
                {"symbol": "BTCUSDT", "name": "BTC", "type": "GLOBAL"},
                {"symbol": "ETHUSDT", "name": "ETH", "type": "GLOBAL"},
                {"symbol": "SOLUSDT", "name": "SOL", "type": "GLOBAL"}
            ]

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
    #  2. HISTORICAL DATA (PHYSICS READY)
    # -------------------------
    @staticmethod
    def get_historical_data(symbol_input, market_type="PERP", trade_style="INTRADAY"):
        # 1. Sanitize Symbol
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
        else:
            raw = str(symbol_input).upper().strip().replace("/", "").replace("-", "")
            symbol = f"{raw}USDT" if "USDT" not in raw else raw

        # 2. Interval Map (Optimized for Physics)
        # Using 15m for Intraday to allow Kinetic Energy calculation
        interval_map = {
            "SCALP": "5m",
            "INTRADAY": "15m",
            "SWING": "4h",
            "POSITION": "1d"
        }
        interval = interval_map.get(trade_style, "15m")

        # 3. Check Cache
        cache_key = f"kline_v85:{symbol}:{interval}:{market_type}"
        df = cache.get(cache_key)

        if df is None or df.empty:
            # 4. Fetch from Binance (Multi-Page for depth)
            base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT
            limit = 1000
            klines = []

            try:
                # Fetch recent data
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                resp = requests.get(f"{base_url}/klines", params=params, timeout=5)

                if resp.status_code == 200:
                    batch = resp.json()
                    if isinstance(batch, list):
                        klines = batch

                if not klines: return pd.DataFrame()

                # 5. Build DataFrame with PHYSICS COLUMNS
                # 0:OpenTime, 1:Open, 2:High, 3:Low, 4:Close, 5:Vol, 6:CloseTime, 7:QuoteVol, 8:Trades, 9:TakerBase
                df = pd.DataFrame(klines, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_base",
                    "taker_quote", "ignore"
                ])

                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                df.set_index("timestamp", inplace=True)

                # Convert to float
                cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "trades", "taker_base"]
                df = df[cols].astype(float)

                # Cache ONLY if sufficient depth
                if len(df) >= 50:
                    cache.set(cache_key, df, timeout=60)  # 1 Minute Cache for Freshness

            except Exception as e:
                log.error(f"Physics Data Fetch Error: {e}")
                return pd.DataFrame()

        # ----------------------------
        # 4. LIVE PRICE INJECTION
        # ----------------------------
        # Even if cached, fetch the REAL-TIME price to fix the "$ --" header bug
        # This ensures the engine always has the exact current price.
        try:
            r = requests.get(
                f"{MarketService.BASE_FUT}/ticker/price",
                params={"symbol": symbol},
                timeout=3
            )
            if r.status_code == 200:
                live_price = float(r.json().get("price"))

                # Inject into DataFrame (copy to avoid cache mutation issues)
                df = df.copy()
                df["live_close"] = df["close"]  # Clone column

                # Update last row with live price
                df.iloc[-1, df.columns.get_loc("close")] = live_price
                df.iloc[-1, df.columns.get_loc("live_close")] = live_price

                # Recalculate last candle Volume/Trades roughly if needed,
                # but Price is the most critical for the UI.
        except Exception:
            pass  # Fallback to kline close if live fetch fails

        return df