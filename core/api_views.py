# core/api_views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from dataclasses import asdict
import logging

from .services.marketdata_service import MarketService
from .quant.quant_engine import run_quant, _get_engine_cls
from .backtest.backtest_engine import BacktestEngine
from .services.market_universe import get_enabled_markets
from .services.search_service import search_symbols
from .services.ai_commentary_service import generate_quant_commentary # New Service
from .models import Feedback

log = logging.getLogger(__name__)

class AnalyzeMarketView(APIView):
    def post(self, request):
        try:
            data = request.data
            symbol = data.get("symbol")
            if not symbol:
                return Response({"error": "Symbol is required"}, status=400)

            market = data.get("market_type", "EQUITY")
            style = data.get("trade_style", "SWING")

            # 1. Fetch Data
            df = MarketService.get_historical_data(
                {"symbol": symbol, "token": data.get("token")},
                market,
                style
            )

            # 2. Fallback for Options/Futures (Use Index)
            if (df is None or df.empty) and market in ["OPTIONS", "FUTURES"]:
                idx = "NIFTY"
                if "BANK" in symbol: idx = "BANKNIFTY"
                elif "FIN" in symbol: idx = "FINNIFTY"
                df = MarketService.get_historical_data({"symbol": idx}, "FUTURES", "INTRADAY")

            if df is None or df.empty:
                return Response({"error": f"Data unavailable for {symbol}."}, status=404)

            # 3. Run Engine
            result = run_quant(market, df, symbol, style)

            # 4. Serialize
            try:
                response_data = asdict(result)
            except Exception:
                response_data = result.__dict__

            # 5. Generate AI Commentary (Pass result object directly)
            response_data["ai_comment"] = generate_quant_commentary(result)

            return Response(response_data)

        except Exception as e:
            log.exception(f"Engine Error: {e}")
            return Response({"error": f"Analysis Failed: {str(e)}"}, status=500)


class BacktestView(APIView):
    def post(self, request):
        try:
            d = request.data
            sym = d.get("symbol")
            mkt = d.get("market_type", "EQUITY")
            style = d.get("trade_style", "SWING")

            # For Backtest, we need MORE data.
            # Force 1D timeframe for long-term backtest if Style is LongTerm
            # Or 15m for Swing.
            # Yahoo limits intraday to 60d. SmartAPI limits to recent history.

            # Strategy: Try fetching max available
            df = MarketService.get_historical_data(
                {"symbol": sym, "token": d.get("token")}, mkt, style
            )

            # Options Backtest Fix: Use Index Data
            if (df is None or len(df) < 50) and mkt in ["OPTIONS", "FUTURES"]:
                idx = "NIFTY"
                if "BANK" in sym: idx = "BANKNIFTY"
                df = MarketService.get_historical_data({"symbol": idx}, "FUTURES", "INTRADAY")

            if df is None or len(df) < 50:
                return Response({"error": "Not enough history for backtest (Need >50 candles)"}, 400)

            engine_cls = _get_engine_cls(mkt)
            bt = BacktestEngine(engine_cls, df, sym, style)

            # Run simulation
            # Start from index 30 to allow indicators to warm up
            res = bt.run(start_idx=30)

            return Response(asdict(res))
        except Exception as e:
            log.error(f"Backtest Error: {e}")
            return Response({"error": str(e)}, 500)

class MarketUniverseView(APIView):
    def get(self, request):
        return Response({"markets": get_enabled_markets()})

class SearchSymbolView(APIView):
    def get(self, request):
        q = request.GET.get("q", "")
        m = request.GET.get("market_type", "EQUITY")
        return Response(search_symbols(q, m))

class SubmitFeedbackView(APIView):
    def post(self, request):
        return Response({"status": "ok"})