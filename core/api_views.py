# core/api_views.py
import csv
import json
import os
import traceback
import logging
import concurrent.futures

from django.core.cache import cache
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated

from reelioo import settings
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
            if not symbol.endswith("USDT") and not symbol.endswith("BTC"):
                symbol += "USDT"

            trade_style = body.get("trade_style", "INTRADAY")
            market_type = body.get("market_type", "PERP")

            df = MarketService.get_historical_data(symbol, market_type, trade_style)

            if df.empty:
                return Response({"error": "No Data or API Error"}, status=400)

            engine = CryptoQuantEngine()
            res = engine.analyze(df, trade_style)

            # ---------------------------------------------------------------
            # ❌ DELETED SAFETY VALVE BLOCK
            # The Engine (v5.4) now handles volatility logic internally.
            # It will correctly return "WATCH" (Yellow) or "HOLD" (Gray)
            # without this external override blocking it.
            # ---------------------------------------------------------------

            news_data = []
            try:
                news_data = NewsService.get_smart_insights(symbol)
            except:
                pass

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
                    "duration": res.expected_duration,
                    "narrative": getattr(res, 'narrative', "Analyzing...") # PASS NARRATIVE TO FRONTEND
                },
                "regime": {"phase": res.regime, "color": res.regime_color},
                "whales": {"zscore": res.whale_zscore, "label": res.whale_label},
                "explainability": res.top_features,
                "sentiment": {"headline": "AI Active", "news_feed": news_data}
            })

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)


class BacktestCryptoView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            body = request.data
            symbol = body.get("symbol", "BTCUSDT").upper()
            if not symbol.endswith("USDT"): symbol += "USDT"
            market_type = body.get("market_type", "PERP")
            trade_style = body.get("trade_style", "INTRADAY")

            df = MarketService.get_historical_data(symbol, market_type, trade_style="INTRADAY")

            if df is None or df.empty:
                return Response({"error": "Insufficient historical data"}, status=404)

            engine = CryptoBacktestEngine(df, symbol)
            results = engine.run(trade_style=trade_style)
            return Response(results)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


class SearchCryptoView(APIView):
    def get(self, request):
        q = request.GET.get("q", "")
        return Response(MarketService.search_assets(q))


class GlobalSymbolsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        csv_path = os.path.join(settings.BASE_DIR, 'global_symbols.csv')
        symbols = []
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'symbol' in row: symbols.append(row['symbol'])
            except Exception as e: pass
        if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        return Response(symbols)


class FindAlphaView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # 1. Check Cache (5 Minutes)
        cached_result = cache.get("alpha_opportunity_v1")
        if cached_result: return Response(cached_result)

        # 2. Define Scan List
        vip_assets = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "WIFUSDT",
            "SUIUSDT", "MATICUSDT", "NEARUSDT", "APTUSDT", "INJUSDT",
            "PEPEUSDT", "RNDRUSDT", "FETUSDT", "LTCUSDT"
        ]

        leaderboard = []
        engine = CryptoQuantEngine()

        def analyze_symbol(symbol):
            try:
                # Fetch Data
                df = MarketService.get_historical_data(symbol, market_type="PERP", trade_style="INTRADAY")
                if df.empty: return None

                # Analyze
                res = engine.analyze(df, trade_style="INTRADAY")

                # --- STRICT SNIPER FILTER ---
                # 1. Must be LONG or SHORT (No WATCH, No HOLD)
                # 2. Score must be >= 70 (Confirmed Strength)
                if res.bias in ["LONG", "SHORT"] and res.score >= 70:
                    return {
                        "symbol": symbol,
                        "bias": res.bias,
                        "score": res.score,
                        "entry": res.entry,
                        "stop": res.stop,
                        "target": res.target1,
                        "rr": res.rr_ratio,
                        "regime": res.regime,
                        "explanation": getattr(res, 'narrative', "High Conviction Setup")
                    }
            except Exception:
                return None
            return None

        # 3. Parallel Execution (Fast Scan)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(analyze_symbol, vip_assets))

        leaderboard = [r for r in results if r is not None]

        # 4. Final Decision
        if not leaderboard:
            # EMPTY RESULT -> Triggers "Capital Preserved" Modal on Frontend
            result = {
                "status": "empty",
                "trade": {
                    "symbol": "MARKET",
                    "bias": "NEUTRAL",
                    "explanation": "No confirmed setups. Capital Preserved.",
                    "entry": 0, "stop": 0, "target": 0
                }
            }
        else:
            # Success -> Show Best Trade
            leaderboard.sort(key=lambda x: x['score'], reverse=True)
            result = {"status": "success", "trade": leaderboard[0]}

        # Cache Result
        cache.set("alpha_opportunity_v1", result, 300)
        return Response(result)