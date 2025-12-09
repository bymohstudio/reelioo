# core/quant/signal_normalizer.py
"""
Signal Normalizer for Reelioo Quant Engine
------------------------------------------

Ensures consistent structure across:
- EQUITY
- FUTURES
- OPTIONS
- CRYPTO

Does NOT override existing data.
It only fills missing keys and normalizes formats so UI never breaks.
"""

import math


def _safe_float(val, default=None):
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _normalize_manipulation_fields(extras: dict):
    """
    Ensure liquidity and stop-run fields are always present.
    """
    extras.setdefault("liquidity_grab_risk", "UNKNOWN")
    extras.setdefault("stop_run_risk", "UNKNOWN")
    extras.setdefault("stop_run_details", {})


def _normalize_options_fields(extras: dict):
    """
    Ensure options-specific fields exist (even if None).
    """
    extras.setdefault("underlying", None)
    extras.setdefault("expiry", None)
    extras.setdefault("strike", None)
    extras.setdefault("option_type", None)

    extras.setdefault("theta_decay_score", None)
    extras.setdefault("iv_crush_probability", None)
    extras.setdefault("gamma_zone", None)

    extras.setdefault("ltp_info", {})
    extras.setdefault("chain_info", {})


def _normalize_equity_fields(extras: dict):
    extras.setdefault("trend", "NEUTRAL")
    extras.setdefault("volatility_regime", "MEDIUM")
    extras.setdefault("ml_edge", None)


def _normalize_futures_fields(extras: dict):
    extras.setdefault("vwap_dev", None)
    extras.setdefault("velocity", None)
    extras.setdefault("trend_score", None)
    extras.setdefault("ml_edge", None)


def _normalize_crypto_fields(extras: dict):
    extras.setdefault("volatility_regime", "MEDIUM")
    extras.setdefault("trend_score", None)
    extras.setdefault("breakout_probability", None)
    extras.setdefault("whale_activity_z", None)
    extras.setdefault("ml_edge", None)


def normalize_quant_result_payload(payload: dict):
    """
    Main entrypoint.
    Called within quant_engine after engine.run(), before returning result to UI.
    """
    try:
        direction = payload.get("direction", "NEUTRAL")
        score = payload.get("score", 0)
        extras = payload.get("extras", {}) or {}

        market_type = extras.get("market_type", "").upper()

        # Always normalize shared manipulation fields
        _normalize_manipulation_fields(extras)

        # Market-specific normalization
        if market_type == "EQUITY":
            _normalize_equity_fields(extras)

        elif market_type == "FUTURES":
            _normalize_futures_fields(extras)

        elif market_type == "OPTIONS":
            _normalize_options_fields(extras)

        elif market_type == "CRYPTO":
            _normalize_crypto_fields(extras)

        # Ensure recommended_timeframe is present
        extras.setdefault("recommended_timeframe", None)
        extras.setdefault("executed_timeframe", None)

        # Safe numeric score
        try:
            payload["score"] = float(score)
        except Exception:
            payload["score"] = 0.0

        payload["direction"] = direction or "NEUTRAL"
        payload["extras"] = extras

        return payload

    except Exception:
        # In worst case, return payload unchanged
        return payload
