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
    # 1. EXCHANGE INFO (SYMBOL DISCOVERY) — v30 UPGRADED
    # =====================================================================
    @staticmethod
    def _load_exchange_info():
        cache_key = "exchange_info_v10"
        cached = cache.get(cache_key)
        if cached:
            log.info("🧠 [EXCHANGE] Loaded symbols from cache")
            return cached

        results = []
        csv_path = os.path.join(settings.BASE_DIR, "global_symbols.csv")

        # ----- CSV FIRST (FAST) -----
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                results = df.to_dict("records")
                log.info(f"✅ [EXCHANGE] Loaded {len(results)} symbols from CSV")
            except Exception as e:
                log.error(f"❌ [EXCHANGE] CSV read failed: {e}")

        # ----- BINANCE FUTURES FALLBACK -----
        if not results:
            try:
                r = requests.get(
                    f"{MarketService.BASE_FUT}/exchangeInfo",
                    timeout=5
                )
                data = r.json()

                for s in data.get("symbols", []):
                    if s["status"] != "TRADING":
                        continue
                    if s["quoteAsset"] != "USDT":
                        continue

                    # VOLUME FILTER (PREVENT DEAD SYMBOLS)
                    # If 'volume' not available here, fetch from ticker
                    vol = 0
                    try:
                        vol_resp = requests.get(
                            f"{MarketService.BASE_FUT}/ticker/24hr",
                            params={"symbol": s["symbol"]},
                            timeout=4
                        )
                        vol_data = vol_resp.json()
                        vol = float(vol_data.get("volume", 0))
                    except:
                        pass

                    if vol < 500000:
                        continue  # Ignore dead markets

                    results.append({
                        "symbol": s["symbol"],
                        "name": s["baseAsset"],
                        "type": "PERP"
                    })

                log.info(f"🌐 [EXCHANGE] Loaded {len(results)} valid symbols from Binance")

            except Exception as e:
                log.error(f"❌ [EXCHANGE] Binance fallback failed: {e}")

        # ----- ABSOLUTE SAFETY NET -----
        if not results:
            results = [
                {"symbol": "BTCUSDT", "name": "BTC", "type": "PERP"},
                {"symbol": "ETHUSDT", "name": "ETH", "type": "PERP"},
                {"symbol": "SOLUSDT", "name": "SOL", "type": "PERP"},
            ]
            log.warning("⚠️ [EXCHANGE] Using static fallback symbols")

        # Cache for 24 hours
        cache.set(cache_key, results, timeout=86400)
        return results

    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query:
            return []
        symbols = MarketService._load_exchange_info()
        return [s for s in symbols if query in s["symbol"] or query in s["name"]][:10]

    # =====================================================================
    # 2. HISTORICAL DATA (PHYSICS-GRADE) — v30 HARDENED
    # =====================================================================
    @staticmethod
    def get_historical_data(symbol_input, market_type="PERP", trade_style="INTRADAY"):
        # -------------------------------------------------
        # 0. SYMBOL NORMALIZATION
        # -------------------------------------------------
        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
        else:
            raw = str(symbol_input).upper().replace("/", "").replace("-", "")
            symbol = raw if raw.endswith("USDT") else f"{raw}USDT"

        log.info(f"📊 [DATA] Fetching for {symbol}")

        # -------------------------------------------------
        # 1. INTERVAL SELECTION
        # -------------------------------------------------
        interval_map = {
            "SCALP": "5m",
            "INTRADAY": "15m",
            "SWING": "1h",
            "POSITION": "4h",
        }
        interval = interval_map.get(trade_style, "15m")

        # -------------------------------------------------
        # 2. CACHE LOOKUP
        # -------------------------------------------------
        cache_key = f"kline_phys_v10:{symbol}:{interval}:{market_type}"
        MIN_ROWS = 2000

        df = cache.get(cache_key)
        if df is not None and isinstance(df, pd.DataFrame) and len(df) >= MIN_ROWS:
            # HARD FAIL CHECK — protect from corrupted cache
            if df["close"].std() == 0 or df["high"].max() == df["low"].min():
                log.error("❌ [CACHE] Corrupted dataframe — invalidating")
                df = None
            else:
                log.info(f"🧠 [CACHE] HIT {symbol} ({len(df)} candles)")

        else:
            df = None

        # -------------------------------------------------
        # 3. MULTI-PAGE HISTORICAL FETCH
        # -------------------------------------------------
        if df is None:
            klines = []
            base_url = (
                MarketService.BASE_FUT
                if market_type in ["PERP", "FUTURES"]
                else MarketService.BASE_SPOT
            )

            limit = 1500
            start_ts = None
            max_rows = 5000

            try:
                while len(klines) < max_rows:
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "limit": limit
                    }
                    if start_ts:
                        params["endTime"] = start_ts

                    r = requests.get(
                        f"{base_url}/klines",
                        params=params,
                        timeout=10
                    )

                    if r.status_code != 200:
                        log.error(f"❌ [BINANCE] Kline Error {r.status_code}")
                        break

                    batch = r.json()
                    if not isinstance(batch, list) or not batch:
                        break

                    klines = batch + klines
                    start_ts = batch[0][0] - 1

                    if len(batch) < limit:
                        break

                if not klines:
                    log.error(f"❌ [DATA] No candles for {symbol}")
                    return pd.DataFrame()

                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades",
                        "taker_base", "taker_quote", "ignore"
                    ]
                )

                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                df.set_index("timestamp", inplace=True)

                df = df[
                    ["open", "high", "low", "close",
                     "volume", "quote_volume", "trades", "taker_base"]
                ].astype(float)

                # HARD FAIL CHECK — BINANCE SOMETIMES RETURNS FLAT CANDLES
                if df["close"].std() == 0:
                    log.error(f"❌ [DATA] Binance returned invalid data for {symbol}")
                    return pd.DataFrame()

                log.info(f"✅ [DATA] Loaded {len(df)} candles")

                if len(df) >= MIN_ROWS:
                    cache.set(cache_key, df, timeout=300)
                    log.info(f"💾 [CACHE] Stored for {symbol}")
                else:
                    log.warning(
                        f"⚠️ [DATA] Only {len(df)} rows — not caching"
                    )

            except Exception as e:
                log.exception(f"🔥 [DATA] Fetch failed: {e}")
                return pd.DataFrame()

        # -------------------------------------------------
        # 4. LIVE PRICE INJECTION
        # -------------------------------------------------
        try:
            r = requests.get(
                f"{MarketService.BASE_FUT}/ticker/price",
                params={"symbol": symbol},
                timeout=3
            )
            live_price = float(r.json().get("price"))
            df = df.copy()
            df["live_close"] = df["close"]
            df.iloc[-1, df.columns.get_loc("live_close")] = live_price
            log.info(f"💰 [PRICE] Live injected: {live_price}")
        except:
            df["live_close"] = df["close"]
            log.warning("⚠️ [PRICE] Live price fallback")

        return df
