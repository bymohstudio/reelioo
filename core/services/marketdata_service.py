import os
import requests
import pandas as pd
import logging
from django.core.cache import cache
from django.conf import settings

log = logging.getLogger(__name__)


class MarketService:
    BASE_SPOT = "https://api.binance.com/api/v3"
    BASE_FUT = "https://fapi.binance.com/fapi/v1"

    # =====================================================================
    # 1. EXCHANGE INFO (OPTIMIZED SYMBOL LOADER)
    # =====================================================================
    @staticmethod
    def _load_exchange_info():
        cache_key = "exchange_info_v36"
        cached = cache.get(cache_key)
        if cached:
            return cached

        results = []

        try:
            info_resp = requests.get(f"{MarketService.BASE_FUT}/exchangeInfo", timeout=8)
            info_data = info_resp.json()
            symbols_raw = info_data.get("symbols", [])
        except Exception as e:
            log.error(f"[EXCHANGE] Info fetch failed: {e}")
            return []

        ticker_map = {}
        try:
            ticker_resp = requests.get(f"{MarketService.BASE_FUT}/ticker/24hr", timeout=8)
            ticker_data = ticker_resp.json()
            for t in ticker_data:
                ticker_map[t['symbol']] = float(t.get('quoteVolume', 0))
        except Exception as e:
            log.error(f"[EXCHANGE] Ticker fetch failed: {e}")

        MIN_USDT_VOL = 5_000_000

        for s in symbols_raw:
            if s["status"] != "TRADING":
                continue
            if s["quoteAsset"] != "USDT":
                continue

            vol_usdt = ticker_map.get(s["symbol"], 0)

            if vol_usdt < MIN_USDT_VOL:
                continue

            results.append({
                "symbol": s["symbol"],
                "name": s["baseAsset"],
                "type": "PERP",
                "vol_24h": vol_usdt
            })

        results.sort(key=lambda x: x['vol_24h'], reverse=True)

        log.info(f"[EXCHANGE] Loaded {len(results)} valid symbols")

        cache.set(cache_key, results, timeout=21600)
        return results

    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query:
            return []
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
            raw = str(symbol_input).upper().replace("/", "").replace("-", "").replace(".P", "")
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
        cache_key = f"kline_v36:{symbol}:{interval}:{market_type}"
        MIN_ROWS = 250

        df = cache.get(cache_key)
        if df is not None and len(df) >= MIN_ROWS:
            # Still inject live price even on cache hit
            try:
                base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT
                ticker_url = f"{base_url}/ticker/price"
                r = requests.get(ticker_url, params={"symbol": symbol}, timeout=2)
                if r.status_code == 200:
                    live_price = float(r.json().get("price"))
                    df.iloc[-1, df.columns.get_loc("close")] = live_price
            except Exception:
                pass
            return df

        # API Setup
        base_url = MarketService.BASE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BASE_SPOT
        limit = 1500 if "fapi" in base_url else 1000

        klines = []
        start_ts = None
        max_rows = 4500

        try:
            while len(klines) < max_rows:
                params = {"symbol": symbol, "interval": interval, "limit": limit}
                if start_ts:
                    params["endTime"] = start_ts

                r = requests.get(f"{base_url}/klines", params=params, timeout=8)

                if r.status_code == 429:
                    log.critical("[DATA] BINANCE RATE LIMIT HIT! COOLING DOWN.")
                    break

                if r.status_code != 200:
                    break

                batch = r.json()
                if not batch:
                    break

                klines = batch + klines
                start_ts = batch[0][0] - 1

                if len(batch) < limit:
                    break

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            cols = ["open", "high", "low", "close", "volume", "quote_volume"]
            df = df[cols].astype(float)

            # Data Validation
            if df["close"].std() == 0 or len(df) < 200:
                log.warning(f"[DATA] Bad data for {symbol}")
                return pd.DataFrame()

            # Cache good data
            if len(df) >= MIN_ROWS:
                cache.set(cache_key, df, timeout=300)

            # Live Price Injection
            try:
                ticker_url = f"{base_url}/ticker/price"
                r = requests.get(ticker_url, params={"symbol": symbol}, timeout=2)
                if r.status_code == 200:
                    live_price = float(r.json().get("price"))
                    df.iloc[-1, df.columns.get_loc("close")] = live_price
            except Exception:
                pass

            return df

        except Exception as e:
            log.error(f"[DATA] Fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    # =====================================================================
    # 3. ORDER FLOW SNAPSHOT
    # =====================================================================
    @staticmethod
    def get_order_book_snapshot(symbol):
        """
        Fetches Level 2 Depth for OBI (Order Book Imbalance) Calculation.
        Used by v36 Engine to detect Institutional Intent.
        """
        clean_sym = str(symbol).upper().replace("/", "").replace("-", "").replace(".P", "")

        try:
            url = f"{MarketService.BASE_FUT}/depth"
            r = requests.get(url, params={"symbol": clean_sym, "limit": 50}, timeout=3)

            if r.status_code != 200:
                return None

            data = r.json()

            # Validate response structure
            if not data.get('bids') or not data.get('asks'):
                return None

            return data

        except Exception as e:
            log.debug(f"[ORDERBOOK] Fetch failed for {clean_sym}: {e}")
            return None