from .services.marketdata_service import MarketService
from .quant.crypto_engine import CryptoQuantEngine
import logging

log = logging.getLogger(__name__)


def analyze_market_data(symbol):
    """
    Standalone function to analyze a symbol and return a dictionary.
    Used by Cron Jobs and Internal Logic (Discord Alerts).
    """
    try:
        # 1. Fetch Data
        # We use INTRADAY (1h) for the standard scanner
        df = MarketService.get_historical_data(symbol, "AUTO", "INTRADAY")

        if df.empty:
            return {}

        # 2. Run The Engine
        engine = CryptoQuantEngine()
        res = engine.analyze(df, "INTRADAY")

        # 3. Format Response
        # We construct a dictionary that matches exactly what the Cron logic expects
        return {
            "symbol": symbol,
            "price": res.entry,
            "signal": {
                "bias": res.bias,
                "probability": res.score,
                "entry": res.entry,
                "stop": res.stop,
                "target2": res.target2,
            },
            "whales": {"zscore": res.whale_zscore, "label": res.whale_label}
        }
    except Exception as e:
        log.error(f"Analysis Error for {symbol}: {e}")
        return {}