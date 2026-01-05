from django.core.cache import cache
import logging

log = logging.getLogger(__name__)

REGIME_TTL = 60 * 15  # 15 minutes memory


def apply_regime_memory(symbol: str, bias: str, score: int):
    """
    Applies stateful decay to confidence to prevent flip-flopping.
    """
    key = f"regime_memory:{symbol}"
    prev = cache.get(key)

    if prev:
        prev_bias = prev["bias"]
        prev_score = prev["score"]

        # If bias unchanged → slight decay
        if bias == prev_bias:
            score = int(prev_score * 0.85 + score * 0.15)
        else:
            # Bias changed → soften transition
            score = int((prev_score + score) / 2)

        log.info(f"🧠 [REGIME] {symbol} memory applied → {prev_bias} → {bias} ({score}%)")

    cache.set(key, {"bias": bias, "score": score}, timeout=REGIME_TTL)
    return score
