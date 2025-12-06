# core/services/marketdata_service.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import pyotp  # Required for TOTP generation

# Try importing Django settings for fallback configuration
try:
    from django.conf import settings
except ImportError:
    settings = None

try:
    from SmartApi import SmartConnect
except ImportError:
    logger.warning("SmartApi not installed. Run 'pip install smartapi-python logzero pyotp'")
    SmartConnect = None

# ==========================================
# 1. PROVIDER INTERFACE
# ==========================================

class MarketDataProvider:
    def get_ohlc(self, symbol_info: dict, interval: str, period_days: int) -> pd.DataFrame:
        raise NotImplementedError

# ==========================================
# 2. PROVIDER: ANGEL ONE (SMART API)
# ==========================================

class SmartApiProvider(MarketDataProvider):
    """
    Handles Indian Equities, Futures, and Options via Angel One.
    FIX: Enforces Session Login (Client ID + Password + TOTP) to resolve AG8001.
    """
    
    def __init__(self):
        self.api_key = os.environ.get("SMART_API_KEY")
        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.password = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP")
        
        self.client = None
        self.session_data = None
        
        # Auto-login on init
        self._login()

    def _login(self):
        if not SmartConnect:
            logger.error("SmartAPI library missing.")
            return

        if not all([self.api_key, self.client_id, self.password, self.totp_key]):
            logger.error("SmartAPI Credentials missing. Check .env for API_KEY, CLIENT_ID, PASSWORD, TOTP.")
            return

        try:
            # 1. Init Client
            self.client = SmartConnect(api_key=self.api_key)
            
            # 2. Generate TOTP
            try:
                totp = pyotp.TOTP(self.totp_key).now()
            except Exception as e:
                logger.error(f"Invalid TOTP Key: {e}")
                return

            # 3. Generate Session (Crucial for AG8001 fix)
            # Use generateSession which handles the login flow internally
            data = self.client.generateSession(self.client_id, self.password, totp)
            
            if data['status']:
                self.session_data = data['data']
                # logger.info(f"SmartAPI Session Created. JWT: {self.session_data.get('jwtToken', '')[:10]}...")
            else:
                logger.error(f"SmartAPI Login Failed: {data['message']} (Code: {data['errorcode']})")
                
        except Exception as e:
            logger.error(f"SmartAPI Login Exception: {e}")

    def get_ohlc(self, symbol_info: dict, interval: str, period_days: int) -> pd.DataFrame:
        # Ensure session is active
        if not self.client or not self.session_data:
            self._login() # Retry login
            if not self.session_data:
                logger.error("SmartAPI: Cannot fetch data without valid session.")
                return pd.DataFrame()

        # Extract & Validate Token
        token = str(symbol_info.get("token", "")).strip() # Force String
        exchange = symbol_info.get("exchange", "NSE")
        
        if not token:
            logger.error("SmartAPI: Token is empty.")
            return pd.DataFrame()

        # Interval Map
        interval_map = {
            "15m": "FIFTEEN_MINUTE",
            "1h": "ONE_HOUR",
            "1d": "ONE_DAY"
        }
        smart_interval = interval_map.get(interval, "ONE_HOUR")
        
        # Date Calculation
        to_date = datetime.now()
        from_date = to_date - timedelta(days=period_days)
        
        try:
            historicParam = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": smart_interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            
            # Fetch
            res = self.client.getCandleData(historicParam)
            
            # Handle Errors
            if not res:
                return pd.DataFrame()
            
            if res.get('status') is False:
                # Log specific error from Angel One
                logger.error(f"SmartAPI Error {res.get('errorcode')}: {res.get('message')} | Params: {historicParam}")
                return pd.DataFrame()

            if res.get('data'):
                data = res['data']
                df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert to numeric
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[cols].apply(pd.to_numeric, errors='coerce')
                
                return df
            else:
                logger.warning(f"SmartAPI: No data returned for {token}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"SmartAPI Data Fetch Exception: {e}")
            return pd.DataFrame()

# ==========================================
# 3. PROVIDER: WAZIRX (CRYPTO)
# ==========================================

class WazirXProvider(MarketDataProvider):
    BASE_URL = "https://api.wazirx.com/sapi/v1/klines"

    def get_ohlc(self, symbol_info: dict, interval: str, period_days: int) -> pd.DataFrame:
        symbol = symbol_info.get("symbol", "").lower()
        if "inr" not in symbol and "usdt" not in symbol:
            symbol += "inr"

        wazir_interval = interval # WazirX matches 15m, 1h, 1d logic mostly
        limit = min(2000, period_days * 24) 

        try:
            params = {"symbol": symbol, "interval": wazir_interval, "limit": limit}
            resp = requests.get(self.BASE_URL, params=params, timeout=5)
            data = resp.json()

            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
                df.set_index("timestamp", inplace=True)
                
                cols = ["Open", "High", "Low", "Close", "Volume"]
                df = df[cols].astype(float)
                return df
                
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"WazirX Error: {e}")
            return pd.DataFrame()

# ==========================================
# 4. MAIN ADAPTER
# ==========================================

class MarketService:
    
    _smart_api = None
    _wazir_api = None

    @classmethod
    def _get_provider(cls, market_type: str):
        mt = market_type.upper().strip()
        if mt == "CRYPTO":
            if not cls._wazir_api: cls._wazir_api = WazirXProvider()
            return cls._wazir_api
        else:
            if not cls._smart_api: cls._smart_api = SmartApiProvider()
            return cls._smart_api

    @classmethod
    def get_historical_data(cls, symbol_info: dict, market_type: str = "EQUITY", trade_style: str = "SWING") -> pd.DataFrame:
        style = (trade_style or "SWING").upper()
        config = {
            "INTRADAY": ("15m", 15),
            "SWING":    ("1h", 90),
            "LONG_TERM": ("1d", 365)
        }
        interval, days = config.get(style, ("1h", 60))
        
        provider = cls._get_provider(market_type)
        return provider.get_ohlc(symbol_info, interval, days)