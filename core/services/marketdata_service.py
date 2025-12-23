import requests
import pandas as pd
import logging
import time
import os
import hmac
import hashlib
import json
from django.core.cache import cache

log = logging.getLogger(__name__)


class MarketService:
    # --- ENDPOINTS ---
    BINANCE_SPOT = "https://api.binance.com/api/v3"
    BINANCE_FUT = "https://fapi.binance.com/fapi/v1"

    COINDCX_PUBLIC = "https://public.coindcx.com"
    COINDCX_API = "https://api.coindcx.com"

    # ---------------------------------------------------------
    #  HELPER: AUTHENTICATION (For Account Data Only)
    # ---------------------------------------------------------
    @staticmethod
    def _get_coindcx_headers(body=None):
        if body is None: body = {}
        key = os.getenv("COINDCX_API_KEY", "")
        secret = os.getenv("COINDCX_SECRET_KEY", "")
        headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}

        # Only add signature for POST requests (when body exists)
        if key and secret and body:
            try:
                timestamp = int(time.time() * 1000)
                body["timestamp"] = timestamp
                json_body = json.dumps(body, separators=(',', ':'))
                signature = hmac.new(secret.encode('utf-8'), json_body.encode('utf-8'), hashlib.sha256).hexdigest()
                headers["X-AUTH-APIKEY"] = key
                headers["X-AUTH-SIGNATURE"] = signature
            except Exception as e:
                log.error(f"Signature Generation Failed: {e}")
        return headers

    # ---------------------------------------------------------
    #  1. EXCHANGE INFO (Global + Indian + Fallback)
    # ---------------------------------------------------------
    @staticmethod
    def _load_exchange_info():
        cache_key = "exchange_info_v17_final"
        cached = cache.get(cache_key)
        if cached: return cached

        results = []

        # --- A. FETCH BINANCE (Global) ---
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            spot = requests.get(f"{MarketService.BINANCE_SPOT}/exchangeInfo", headers=headers, timeout=3).json()
            if "symbols" in spot:
                for s in spot["symbols"]:
                    if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                        results.append({
                            "symbol": s["symbol"],
                            "name": s["baseAsset"],
                            "type": "GLOBAL",
                            "display": f"{s['symbol']} (Global)"
                        })
        except Exception as e:
            log.error(f"Binance Info Error: {e}")

        # --- B. FETCH COINDCX (Indian) ---
        indian_found = False
        try:
            # Public GET for markets (No Auth)
            url = f"{MarketService.COINDCX_API}/exchange/v1/markets_details"
            cdx = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5).json()

            if isinstance(cdx, list):
                for m in cdx:
                    if m.get("target_currency_short_name") == "INR" and m.get("status") == "active":
                        base = m.get("base_currency_short_name")
                        pair_code = m.get("pair")
                        simple_symbol = f"{base}INR".upper()

                        results.append({
                            "symbol": simple_symbol,
                            "name": base,
                            "type": "INDIAN",
                            "cdx_pair": pair_code,
                            "display": f"{simple_symbol} (Indian)"
                        })
                        indian_found = True
        except Exception as e:
            log.error(f"CoinDCX API Failed: {e}")

        # --- C. STATIC FALLBACK (Safety Net) ---
        if not indian_found:
            fallback_coins = ["BTC", "ETH", "SOL", "DOGE", "XRP", "MATIC", "SHIB", "ADA", "BNB", "TRX"]
            for base in fallback_coins:
                simple_symbol = f"{base}INR"
                # Default I- prefix per your script
                results.append({
                    "symbol": simple_symbol,
                    "name": base,
                    "type": "INDIAN",
                    "cdx_pair": f"I-{base}_INR",
                    "display": f"{simple_symbol} (Indian)"
                })

        cache.set(cache_key, results, timeout=600)
        return results

    @staticmethod
    def search_assets(query: str):
        query = (query or "").upper().strip()
        if not query: return []
        all_symbols = MarketService._load_exchange_info()
        matches = [s for s in all_symbols if query in s["symbol"]]
        if len(matches) < 20:
            name_matches = [s for s in all_symbols if query in s["name"] and s not in matches]
            matches.extend(name_matches)
        return matches[:20]

    # ---------------------------------------------------------
    #  3. HISTORICAL DATA ROUTER
    # ---------------------------------------------------------
    @staticmethod
    def get_historical_data(symbol_input, market_type="SPOT", trade_style="SWING"):
        symbol = ""
        is_indian = False
        cdx_pair = None

        if isinstance(symbol_input, dict):
            symbol = symbol_input.get("symbol")
            is_indian = symbol_input.get("type") == "INDIAN"
            cdx_pair = symbol_input.get("cdx_pair")
        else:
            raw = str(symbol_input).upper().strip().replace("/", "").replace("-", "")
            symbol = raw
            if symbol.endswith("INR"):
                is_indian = True
                all_syms = MarketService._load_exchange_info()
                found = next((x for x in all_syms if x["symbol"] == symbol), None)
                if found:
                    cdx_pair = found["cdx_pair"]
                else:
                    # Fallback
                    base = symbol.replace("INR", "")
                    cdx_pair = f"I-{base}_INR"

        if is_indian:
            return MarketService._fetch_coindcx_candles(symbol, cdx_pair, trade_style)
        else:
            if "USDT" not in symbol and not symbol.endswith("BTC"): symbol += "USDT"
            return MarketService._fetch_binance_candles(symbol, market_type, trade_style)

    # --- BINANCE DRIVER ---
    @staticmethod
    def _fetch_binance_candles(symbol, market_type, trade_style):
        interval_map = {"SCALP": "15m", "INTRADAY": "1h", "SWING": "4h", "POSITION": "1d"}
        interval = interval_map.get(trade_style, "1h")
        limit = 1000
        base_url = MarketService.BINANCE_FUT if market_type in ["PERP", "FUTURES"] else MarketService.BINANCE_SPOT

        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(f"{base_url}/klines", params=params, headers=headers, timeout=5)

            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json()
            if not isinstance(data, list): return pd.DataFrame()

            # GLOBAL DATA PRESERVATION
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "taker_base", "taker_quote", "ignore"
            ])
            cols = ["open", "high", "low", "close", "volume", "taker_base"]
            df[cols] = df[cols].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # 1. GLOBAL: RETURN REAL 'taker_base'
            return df[["open", "high", "low", "close", "volume", "taker_base"]]

        except Exception as e:
            log.error(f"Binance Error: {e}")
            return pd.DataFrame()

    # --- COINDCX DRIVER (Looping Logic) ---
    @staticmethod
    def _fetch_coindcx_candles(symbol, cdx_pair, trade_style):
        interval_map = {"SCALP": "15m", "INTRADAY": "1h", "SWING": "4h", "POSITION": "1d"}
        interval = interval_map.get(trade_style, "1h")
        limit = 1000  # Required for AI

        if not cdx_pair: cdx_pair = f"I-{symbol.replace('INR', '')}_INR"

        all_data = []
        next_end_time = None
        url = f"{MarketService.COINDCX_PUBLIC}/market_data/candles"

        print(f"🇮🇳 Fetching CoinDCX: {cdx_pair} [{interval}]")

        # LOOP TO GET 1000 CANDLES
        while len(all_data) < limit:
            try:
                params = {"pair": cdx_pair, "interval": interval, "limit": 500}
                if next_end_time: params["endTime"] = int(next_end_time)

                # No Auth for Candles
                resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)

                if resp.status_code != 200:
                    # Retry with B- prefix if I- fails (Common issue)
                    if "I-" in cdx_pair and len(all_data) == 0:
                        cdx_pair = cdx_pair.replace("I-", "B-")
                        print(f"⚠️ Retrying with: {cdx_pair}")
                        continue
                    break

                data = resp.json()
                if not isinstance(data, list) or not data: break

                all_data.extend(data)
                oldest_time = min([d['time'] for d in data])
                next_end_time = oldest_time - 1
                time.sleep(0.2)

            except Exception as e:
                log.error(f"CoinDCX Loop Error: {e}")
                break

        if not all_data:
            print(f"⚠️ No data found for {cdx_pair}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=['time'])
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()

        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)

        # 2. INDIAN: SYNTHESIZE 'taker_base'
        df["taker_base"] = df["volume"] * 0.5

        final_df = df.iloc[-limit:]
        print(f"✅ Data Success: {len(final_df)} candles")

        return final_df[["open", "high", "low", "close", "volume", "taker_base"]]