# core/api_views.py
import csv
import json
import os
import traceback
import logging
import concurrent.futures
from datetime import timedelta

from django.core.cache import cache
from django.utils import timezone
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
            # --- SECURITY FIX ---
            user = request.user
            is_premium = user.profile.is_premium if hasattr(user, 'profile') else False
            joined_date = user.date_joined
            trial_end = joined_date + timedelta(days=21)

            if not is_premium and timezone.now() > trial_end:
                return Response({
                    "error": "Trial Expired. Institutional Access Revoked.",
                    "redirect": "/pricing/"
                }, status=403)
            # --------------------

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

            news_data = []
            try:
                news_data = NewsService.get_smart_insights(symbol)
            except:
                pass

            return Response({
                "symbol": symbol,
                "price": res.price,  # <--- FIXED: MAPS TO MARKET PRICE
                "signal": {
                    "bias": res.bias,
                    "probability": res.score,
                    "style": trade_style,
                    "entry": res.entry, # <--- FIXED: MAPS TO TRADE ENTRY
                    "stop": res.stop,
                    "target1": res.target1,
                    "target2": res.target2,
                    "target3": res.target3,
                    "rr": res.rr_ratio,
                    "duration": res.expected_duration,
                    "narrative": res.narrative
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
            user = request.user
            is_premium = user.profile.is_premium if hasattr(user, 'profile') else False
            trial_end = user.date_joined + timedelta(days=21)

            if not is_premium and timezone.now() > trial_end:
                return Response({"error": "Trial Expired"}, status=403)

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
            except Exception as e:
                pass
        if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        return Response(symbols)


class FindAlphaView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        is_premium = user.profile.is_premium if hasattr(user, 'profile') else False
        joined_date = user.date_joined
        trial_end = joined_date + timedelta(days=21)

        if not is_premium and timezone.now() > trial_end:
            return Response({
                "status": "error",
                "message": "Trial Expired. Upgrade to access Alpha Scanner.",
                "redirect": "/pricing/"
            }, status=403)

        cached_result = cache.get("alpha_opportunity_v1")
        if cached_result: return Response(cached_result)

        vip_assets = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "WIFUSDT",
            "SUIUSDT", "MATICUSDT", "NEARUSDT", "APTUSDT", "INJUSDT",
            "RNDRUSDT", "FETUSDT", "LTCUSDT"
        ]

        leaderboard = []
        engine = CryptoQuantEngine()

        def analyze_symbol(symbol):
            try:
                df = MarketService.get_historical_data(symbol, market_type="PERP", trade_style="INTRADAY")
                if df.empty: return None
                res = engine.analyze(df, trade_style="INTRADAY")

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(analyze_symbol, vip_assets))

        leaderboard = [r for r in results if r is not None]

        if not leaderboard:
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
            leaderboard.sort(key=lambda x: x['score'], reverse=True)
            result = {"status": "success", "trade": leaderboard[0]}

        cache.set("alpha_opportunity_v1", result, 300)
        return Response(result)