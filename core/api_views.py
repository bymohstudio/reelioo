# core/api_views.py

import json
import traceback
import logging
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated

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

            # --- CRITICAL FIX: DEFAULT TO PERP (FUTURES) ---
            # Your "Red Pill" logic relies on Taker Buy Volume, which is only
            # accurate in Futures data. "SPOT" data causes the 0% confidence bug.
            market_type = body.get("market_type", "PERP")

            # Fetch Data
            df = MarketService.get_historical_data(symbol, market_type, trade_style)

            if df.empty:
                return Response({"error": "No Data or API Error"}, status=400)

            # Run Engine
            engine = CryptoQuantEngine()
            res = engine.analyze(df, trade_style)

            # Volatility Filter (Safety Valve)
            if trade_style == "SCALP":
                last_open = float(df['open'].iloc[-1])
                last_close = float(df['close'].iloc[-1])
                move_pct = abs(last_close - last_open) / last_close
                if move_pct < 0.002:
                    res.bias = "NEUTRAL"
                    res.regime = "LOW VOLATILITY"
                    res.regime_color = "gray"
                    res.score = 50

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
                    "duration": res.expected_duration
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

            # FIX: Backtest should also use PERP by default for consistency
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


class FindAlphaView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # 1. THE VIP LIST (Don't scan trash, scan volume)
            vip_assets = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "WIFUSDT",
                "SUIUSDT", "MATICUSDT", "NEARUSDT", "APTUSDT", "INJUSDT"
            ]

            leaderboard = []
            engine = CryptoQuantEngine()

            # 2. FAST SCAN LOOP
            for symbol in vip_assets:
                # Reuse your existing MarketService (Standardized Data)
                df = MarketService.get_historical_data(symbol, market_type="PERP", trade_style="INTRADAY")

                if df.empty: continue

                # Run the Brain
                res = engine.analyze(df, trade_style="INTRADAY")

                # 3. FILTER: Only "Actionable" Signals
                # We want High Score (>60) AND a clear direction (LONG/SHORT)
                if res.bias in ["LONG", "SHORT"] and res.score >= 60:
                    leaderboard.append({
                        "symbol": symbol,
                        "bias": res.bias,
                        "score": res.score,
                        "entry": res.entry,
                        "stop": res.stop,
                        "target": res.target1,  # Conservative target
                        "rr": res.rr_ratio,
                        "regime": res.regime,
                        "explanation": res.top_features[0]['desc'] if res.top_features else "TREND ALIGNMENT"
                    })

            # 4. PICK THE WINNER
            if not leaderboard:
                return Response({"status": "empty", "message": "Market is choppy. No high-probability setups found."})

            # Sort by Score (Highest Confidence First)
            leaderboard.sort(key=lambda x: x['score'], reverse=True)
            best_trade = leaderboard[0]

            return Response({
                "status": "success",
                "trade": best_trade
            })

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)