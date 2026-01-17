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

    # =====================================================================
    # 1. EXCHANGE INFO (OPTIMIZED SYMBOL LOADER)
    # =====================================================================
    @staticmethod
    def _load_exchange_info():
        cache_key = "exchange_info_v30"
        cached = cache.get(cache_key)
        if cached:
            log.info("🧠 [EXCHANGE] Loaded symbols from cache")
            return cached

        results = []

        # 1. Fetch Exchange Info (The List)
        try:
            info_resp = requests.get(f"{MarketService.BASE_FUT}/exchangeInfo", timeout=5)
            info_data = info_resp.json()
            symbols_raw = info_data.get("symbols", [])
        except Exception as e:
            log.error(f"❌ [EXCHANGE] Info fetch failed: {e}")
            return []

        # 2. Fetch 24hr Ticker Stats (The Volume) - SINGLE REQUEST
        # This prevents the "Death Loop" of 300+ requests
        ticker_map = {}
        try:
            ticker_resp = requests.get(f"{MarketService.BASE_FUT}/ticker/24hr", timeout=5)
            ticker_data = ticker_resp.json()
            # Map symbol -> quoteVolume (USDT Value)
            for t in ticker_data:
                ticker_map[t['symbol']] = float(t.get('quoteVolume', 0))
        except Exception as e:
            log.error(f"❌ [EXCHANGE] Ticker fetch failed: {e}")

        # 3. Filter & Merge
        # Minimum 5 Million USDT daily volume to ensure liquidity
        MIN_USDT_VOL = 5_000_000

        for s in symbols_raw:
            if s["status"] != "TRADING": continue
            if s["quoteAsset"] != "USDT": continue

            # Use mapped volume (Fast)
            vol_usdt = ticker_map.get(s["symbol"], 0)

            if vol_usdt < MIN_USDT_VOL:
                continue

            results.append({
                "symbol": s["symbol"],
                "name": s["baseAsset"],
                "type": "PERP",
                "vol_24h": vol_usdt
            })

        # Sort by Volume (Highest first) -> Prioritizes Liquid Markets
        results.sort(key=lambda x: x['vol_24h'], reverse=True)

        log.info(f"🌐 [EXCHANGE] Loaded {len(results)} valid symbols (Top Vol: {results[0]['symbol']})")

        # Cache for 6 hours
        cache.set(cache_key, results, timeout=21600)
        return results

    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query: return []
        symbols = MarketService._load_exchange_info()
        return [s for s in symbols if query in s["symbol"] or query in s["name"]][:10]

    # =====================================================================
    # 2. HISTORICAL DATA (ROBUST FETCH)
    # =====================================================================
    @staticmethod
    def get_historical_data(symbol_input, market_type="PERP", trade_style="INTRADAY"):
        # Normalize Symbol
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
        else:
            raw = str(symbol_input).upper().replace("/", "").replace("-", "")
            symbol = raw if raw.endswith("USDT") else f"{raw}USDT"

        # Interval Map
        interval_map = {
            "SCALP": "5m",
            "INTRADAY": "15m",
            "SWING": "1h",
            "POSITION": "4h",
        }
        interval = interval_map.get(trade_style, "15m")

        # Cache Check
        cache_key = f"kline_v30:{symbol}:{interval}:{market_type}"
        MIN_ROWS = 2000  # Need deeper history for EMA 50 / Mass Index

        df = cache.get(cache_key)
        if df is not None and len(df) >= MIN_ROWS:
            return df

        # API Setup
        base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT

        # [FIX] Spot limit is 1000, Futures is 1500
        limit = 1500 if "fapi" in base_url else 1000

        klines = []
        start_ts = None
        max_rows = 4500  # Optimized for Railway RAM

        try:
            while len(klines) < max_rows:
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                if start_ts: params["endTime"] = start_ts

                r = requests.get(f"{base_url}/klines", params=params, timeout=5)

                if r.status_code == 429:  # Rate Limit Hit
                    log.critical("🔥 [DATA] BINANCE RATE LIMIT HIT! COOLING DOWN.")
                    break

                if r.status_code != 200: break

                batch = r.json()
                if not batch: break

                klines = batch + klines  # Prepend older data
                start_ts = batch[0][0] - 1  # Move cursor back

                if len(batch) < limit: break  # Reached beginning of time

            if not klines: return pd.DataFrame()

            # DataFrame Construction
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            cols = ["open", "high", "low", "close", "volume", "quote_volume"]
            df = df[cols].astype(float)

            # [FIX] Data Validation (Flat Candle Check)
            if df["close"].std() == 0 or len(df) < 200:
                log.warning(f"⚠️ [DATA] Bad data for {symbol}")
                return pd.DataFrame()

            # Cache good data
            if len(df) >= MIN_ROWS:
                cache.set(cache_key, df, timeout=300)

            # Live Injection (Optional but recommended)
            try:
                ticker_url = f"{base_url}/ticker/price"
                r = requests.get(ticker_url, params={"symbol": symbol}, timeout=2)
                live_price = float(r.json().get("price"))
                # Update last close to live price for real-time signal
                df.iloc[-1, df.columns.get_loc("close")] = live_price
            except:
                pass

            return df

        except Exception as e:
            log.error(f"❌ [DATA] Fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    # =====================================================================
    # 3. ORDER FLOW SNAPSHOT (NEW FOR v32 ENGINE)
    # =====================================================================
    @staticmethod
    def get_order_book_snapshot(symbol):
        """
        Fetches Level 2 Depth for OBI (Order Book Imbalance) Calculation.
        Used by v32 Engine to block trades into walls.
        """
        try:
            url = f"{MarketService.BASE_FUT}/depth"
            # Limit 50 is enough for "Immediate Walls" and is lighter on API
            r = requests.get(url, params={"symbol": symbol, "limit": 50}, timeout=2)
            if r.status_code != 200: return None
            return r.json()
        except Exception:
            return None