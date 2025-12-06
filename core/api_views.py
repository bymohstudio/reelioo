# core/api_views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from dataclasses import asdict
import traceback
import json

from .models import Prediction, Feedback
from .services.marketdata_service import MarketService
from .services.ai_commentary_service import AICommentary
from .services.search_service import search_symbols

# Import Engines (Indian Market Context)
from .quant.equity_engine import EquityQuantEngine
from .quant.crypto_engine import CryptoQuantEngine
from .quant.futures_engine import FuturesQuantEngine
from .quant.options_engine import OptionsQuantEngine
from .backtest.backtest_engine import BacktestEngine

class AnalyzeMarketView(APIView):
    """
    Unified Router for Indian Markets (SmartAPI) and Crypto (WazirX).
    Expects: symbol, token, exchange, market_type, trade_style.
    """
    def post(self, request):
        try:
            # 1. Extract Params
            symbol = request.data.get("symbol")
            token = request.data.get("token")         # Critical for SmartAPI
            exchange = request.data.get("exchange")   # Critical for SmartAPI
            trade_style = request.data.get("trade_style", "SWING")
            market_type = request.data.get("market_type", "EQUITY").upper()
            
            if not symbol: 
                return Response({"error": "Symbol is required"}, status=400)
            
            # For Equity/Futures/Options, Token is mandatory (unless it's Crypto)
            if market_type != "CRYPTO" and not token:
                return Response({"error": "Token missing. Please select from search dropdown."}, status=400)

            # 2. Fetch Data
            # Pack symbol info for the MarketService adapter
            symbol_info = {
                "symbol": symbol,
                "token": token,
                "exchange": exchange
            }
            
            df = MarketService.get_historical_data(symbol_info, market_type, trade_style)
            
            if df.empty: 
                return Response({
                    "error": f"No data found for {symbol}. Market might be closed or token expired. Please check your SmartAPI credentials."
                }, status=404)

            # 3. Select Engine
            if market_type == "FUTURES":
                result = FuturesQuantEngine.run(df, symbol, trade_style)
                ai_html = AICommentary.generate_futures_commentary(symbol, result.__dict__)
            elif market_type == "OPTIONS":
                result = OptionsQuantEngine.run(df, symbol, trade_style)
                ai_html = AICommentary.generate_options_commentary(symbol, result.__dict__)
            elif market_type == "CRYPTO":
                result = CryptoQuantEngine.run(df, symbol, trade_style)
                ai_html = AICommentary.generate_crypto_commentary(symbol, result.__dict__)
            else:
                # Default to Equity (NSE/BSE)
                result = EquityQuantEngine.run(df, symbol, trade_style)
                ai_html = AICommentary.generate_equity_commentary(symbol, result.__dict__)
                
            # 4. Prepare Response
            data = asdict(result)
            data["ai_comment"] = ai_html
            
            # 5. Save Prediction (Fire and forget)
            self.save_prediction(data, market_type)
            
            return Response(data)
            
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

    def save_prediction(self, data, market_type):
        try:
            Prediction.objects.create(
                symbol=data.get("symbol"),
                market=market_type,
                trade_type=data.get("time_frame", "SWING"),
                model_probability=data.get("score", 0),
                predicted_direction=data.get("direction", "FLAT"),
                entry_price=data.get("entry", 0),
                target_price=data.get("target", 0),
                stop_loss_price=data.get("stop", 0)
            )
        except Exception as e:
            print(f"DB Save Error: {e}") 

class BacktestView(APIView):
    """
    Backtest runner. Updated to accept Token/Exchange for Indian markets.
    """
    def post(self, request):
        try:
            symbol = request.data.get("symbol")
            token = request.data.get("token")
            exchange = request.data.get("exchange")
            market_type = request.data.get("market_type", "EQUITY").upper()
            trade_style = request.data.get("trade_style", "SWING")
            
            if not symbol: return Response({"error": "Symbol required"}, status=400)

            # Pack info
            symbol_info = {"symbol": symbol, "token": token, "exchange": exchange}

            # Fetch Long Term Data for Backtest
            df = MarketService.get_historical_data(symbol_info, market_type, "LONG_TERM")
            
            if len(df) < 50:
                return Response({"error": "Insufficient history for backtest (need >50 candles)"}, status=400)

            # Select Engine Class
            engine_map = {
                "CRYPTO": CryptoQuantEngine, 
                "FUTURES": FuturesQuantEngine,
                "OPTIONS": OptionsQuantEngine, 
                "EQUITY": EquityQuantEngine
            }
            engine_cls = engine_map.get(market_type, EquityQuantEngine)
            
            # Run Simulation
            bt_engine = BacktestEngine(engine_cls, df, symbol, trade_style)
            results = bt_engine.run(start_idx=50)
            
            return Response(asdict(results))
        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

class SearchSymbolView(APIView):
    def get(self, request):
        q = request.GET.get("q", "")
        market_type = request.GET.get("market_type", "EQUITY")
        # Uses the new CSV-based search service
        results = search_symbols(q, market_type)
        return Response(results)

class SubmitFeedbackView(APIView):
    def post(self, request):
        try:
            data = request.data
            Feedback.objects.create(
                market=data.get("market", "GLOBAL"),
                symbol=data.get("symbol", ""),
                accuracy_rating=int(data.get("accuracy_rating", 0)),
                ux_rating=int(data.get("ux_rating", 0)),
                speed_rating=int(data.get("speed_rating", 0)),
                comment=data.get("comment", ""),
                hallucination_report=data.get("hallucination_flag", False)
            )
            return Response({"status": "success"})
        except Exception as e:
            return Response({"error": str(e)}, status=400)