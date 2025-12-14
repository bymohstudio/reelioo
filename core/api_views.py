import json
import traceback
import logging
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated

# --- IMPORTS ---
from .services.marketdata_service import MarketService
from .services.news_service import NewsService
from .quant.crypto_engine import CryptoQuantEngine
from .backtest.backtest_engine import CryptoBacktestEngine

log = logging.getLogger(__name__)


class AnalyzeCryptoView(APIView):
    """
    Main Analysis Endpoint (The "Scan" Button).
    Protected: Only logged-in users can access.
    """
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # FIX: Use DRF's request.data instead of parsing raw body
            body = request.data

            # 1. Clean & Parse Inputs
            # Defaults to BTCUSDT only if input is missing. Otherwise uses USER INPUT.
            raw_symbol = body.get("symbol", "BTCUSDT")
            symbol = raw_symbol.upper().replace("/", "").replace("-", "").replace("_", "")

            trade_style = body.get("trade_style", "SWING")
            market_type = body.get("market_type", "AUTO")  # SPOT or PERP

            # 2. Fetch Market Data (Binance)
            df = MarketService.get_historical_data(symbol, market_type, trade_style)

            if df is None or df.empty:
                return Response({"error": f"No market data found for {symbol}"}, status=400)

            # 3. AI Insights (OpenAI)
            # Replaces the old 'get_news' with the new GPT-4 Engine
            ai_intel = NewsService.get_smart_insights(symbol)

            # 4. Run Quant Engine (XGBoost Analysis)
            engine = CryptoQuantEngine()
            result = engine.analyze(df, trade_style)

            # 5. Build Response
            response = {
                "symbol": symbol,
                "price": float(df["close"].iloc[-1]),
                "signal": {
                    "bias": result.bias,
                    "probability": int(result.score),
                    "style": trade_style,
                    "entry": result.entry,
                    "target1": result.target1,
                    "target2": result.target2,
                    "target3": result.target3,
                    "stop": result.stop,
                    "rr": result.rr_ratio,
                    "duration": result.expected_duration
                },
                "regime": {
                    "phase": result.regime,
                    "color": result.regime_color
                },
                "whales": {
                    "zscore": result.whale_zscore,
                    "label": result.whale_label
                },
                "sentiment": {
                    "headline": "AI Neural Analysis",
                    "news_feed": ai_intel  # Sends OpenAI bullets to UI
                },
                "explainability": result.top_features
            }
            return Response(response)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class BacktestCryptoView(APIView):
    """
    Backtest Endpoint (The "Validate Strategy" Button).
    """
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # FIX: Use DRF's request.data
            body = request.data

            # Fully Dynamic: Uses whatever symbol is sent from frontend
            symbol = body.get("symbol", "BTCUSDT").upper()

            # We force INTRADAY data for backtesting accuracy
            # But we allow market_type to be passed (e.g. PERP backtesting)
            market_type = body.get("market_type", "SPOT")

            df = MarketService.get_historical_data(symbol, market_type, trade_style="INTRADAY")

            if df is None or df.empty:
                return Response({"error": "Insufficient historical data for backtest"}, status=404)

            engine = CryptoBacktestEngine(df, symbol)
            results = engine.run()

            return Response(results)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class SearchCryptoView(APIView):
    """
    Autosuggest Endpoint.
    Publicly accessible to allow smooth UX before hitting enter.
    """

    def get(self, request):
        q = request.GET.get("q", "")
        return Response(MarketService.search_assets(q))