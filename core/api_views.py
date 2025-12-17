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
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            body = request.data
            symbol = body.get("symbol", "BTCUSDT").upper().replace("-", "")
            trade_style = body.get("trade_style", "INTRADAY")

            # 1. Fetch Data (Timeframe handled by Service)
            df = MarketService.get_historical_data(symbol, "AUTO", trade_style)

            if df.empty:
                return Response({"error": "No Data"}, status=400)

            # 2. Run Engine
            engine = CryptoQuantEngine()
            res = engine.analyze(df, trade_style)

            # 3. Volatility Filter (Prevent Scalping in dead markets)
            if trade_style == "SCALP":
                last_open = float(df['open'].iloc[-1])
                last_close = float(df['close'].iloc[-1])
                move_pct = abs(last_close - last_open) / last_close
                if move_pct < 0.002:  # < 0.2% movement
                    res.bias = "NEUTRAL"
                    res.regime = "LOW VOLATILITY"
                    res.regime_color = "gray"
                    res.score = 50

            # 4. Response
            return Response({
                "symbol": symbol,
                "price": res.entry,
                "signal": {
                    "bias": res.bias,
                    "probability": res.score,
                    "style": trade_style,
                    "entry": res.entry,
                    "stop": res.stop,
                    "target1": res.target1,
                    "target2": res.target2,
                    "target3": res.target3,
                    "rr": res.rr_ratio,
                    "duration": res.expected_duration
                },
                "regime": {"phase": res.regime, "color": res.regime_color},
                "whales": {"zscore": res.whale_zscore, "label": res.whale_label},
                # Optional: Fetch News
                "sentiment": {"headline": "AI Active", "news_feed": []}
            })

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class BacktestCryptoView(APIView):
    """
    Backtest Endpoint.
    """
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            body = request.data
            symbol = body.get("symbol", "BTCUSDT").upper()
            market_type = body.get("market_type", "SPOT")

            # Always backtest on 1H data for speed/accuracy balance
            df = MarketService.get_historical_data(symbol, market_type, trade_style="INTRADAY")

            if df is None or df.empty:
                return Response({"error": "Insufficient historical data"}, status=404)

            engine = CryptoBacktestEngine(df, symbol)
            results = engine.run()

            return Response(results)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


class SearchCryptoView(APIView):
    def get(self, request):
        q = request.GET.get("q", "")
        return Response(MarketService.search_assets(q))