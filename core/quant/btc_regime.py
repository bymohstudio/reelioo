from django.core.cache import cache
import logging

log = logging.getLogger(__name__)

BTC_KEY = "btc_global_regime"
BTC_TTL = 60 * 5  # 5 minutes


def update_btc_regime(bias: str, score: int):
    """
    Called when BTCUSDT is analyzed.
    """
    regime = {
        "bias": bias,
        "score": score
    }
    cache.set(BTC_KEY, regime, timeout=BTC_TTL)
    log.info(f"🟠 [BTC REGIME] Updated → {bias} ({score}%)")


def apply_btc_gating(symbol: str, bias: str, score: int):
    """
    Applies BTC-first gating to altcoins.
    """
    if symbol.startswith("BTC"):
        return bias, score

    btc = cache.get(BTC_KEY)
    if not btc:
        return bias, score

    btc_bias = btc["bias"]
    btc_score = btc["score"]

    # Hard risk-off conditions
    if btc_bias in ["HOLD"] and btc_score < 55:
        log.info(f"⛔ [BTC GATE] {symbol} blocked (BTC neutral)")
        return "HOLD", 50

    if btc_bias == "SHORT":
        log.info(f"⛔ [BTC GATE] {symbol} blocked (BTC bearish)")
        return "HOLD", 50

    # Soft downgrade
    if btc_bias == "WATCH" and bias == "LONG":
        log.info(f"⚠️ [BTC GATE] {symbol} downgraded to WATCH")
        return "WATCH", min(score, 60)

    return bias, score
