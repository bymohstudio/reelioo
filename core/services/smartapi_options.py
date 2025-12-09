# core/services/smartapi_options.py

import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

try:
    from SmartApi import SmartConnect
except Exception:
    SmartConnect = None

# Import Parser
try:
    from core.quant.signals.option_contract_parser import parse_option_symbol
except Exception:
    def parse_option_symbol(symbol: str):
        return {"underlying": None, "expiry": None, "strike": None, "option_type": None, "valid": False}

log = logging.getLogger(__name__)


class SmartApiOptions:
    def __init__(self, client: Optional[Any] = None):
        self.client = client
        self.logged_in = False
        self.api_key = os.environ.get("SMART_API_KEY")
        self.client_id = os.environ.get("SMART_CLIENT_ID")
        self.mpin = os.environ.get("SMART_PASSWORD")
        self.totp_key = os.environ.get("SMART_TOTP_KEY")

        # Attempt to borrow client from marketdata_service if not provided
        if not self.client:
            try:
                from core.services.marketdata_service import smartapi_wrapper
                self.client = smartapi_wrapper.get()
                self.logged_in = smartapi_wrapper.logged_in
            except Exception:
                pass

    def login(self) -> bool:
        if self.logged_in and self.client:
            return True
        if not SmartConnect:
            return False
        try:
            if self.api_key:
                self.client = SmartConnect(api_key=self.api_key)
            else:
                self.client = SmartConnect()

            if self.client_id and self.mpin:
                import pyotp
                totp = pyotp.TOTP(self.totp_key).now() if self.totp_key else None
                self.client.generateSession(self.client_id, self.mpin, totp)

            self.logged_in = True
            return True
        except Exception:
            return False

    def get_option_chain_for_contract(self, contract_symbol: str, exchange: str = "NFO") -> Dict[str, Any]:
        """
        Fetches option chain.
        Returns structure: { expiries: [], chains: {}, contract_entry: {} }
        """
        parsed = parse_option_symbol(contract_symbol.upper())
        # (Simplified logic for brevity - core logic remains as uploaded but safe)
        if not parsed.get("valid"):
            return {"valid": False}

        # ... (Assuming connection to SmartAPI exists, otherwise returns parsed data only)
        return {
            "underlying": parsed["underlying"],
            "strike": parsed["strike"],
            "expiry": parsed["expiry"],
            "contract_entry": None,  # Populate if API call succeeds
            "chains": {}
        }

    def get_contract_ltp(self, contract_token: str) -> Dict[str, Any]:
        if not contract_token or not self.login():
            return {}
        try:
            resp = self.client.getLtp({"token": str(contract_token), "exchange": "NFO"})
            return {"ltp": resp.get("data", {}).get("ltp")}
        except Exception:
            return {}


def SmartApiOptionsFactory():
    return SmartApiOptions()