# core/api_views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from dataclasses import asdict
import traceback

from .services.marketdata_service import MarketService
from .quant.quant_engine import run_quant
from .backtest.backtest_engine import BacktestEngine
from .services.market_universe import get_enabled_markets
from .services.search_service import search_symbols
from .models import Feedback


# --- HELPER: GENERATE RICH AI COMMENTARY ---
def generate_quant_commentary(result) -> str:
    """
    Generates a 'Rich Text' HTML Desk Note with visual pills and structure.
    """
    score = result.score
    direction = result.direction
    vol = result.volatility_regime
    market = result.market_type

    # 1. Sentiment Pill
    if score >= 55:
        badge_cls = "bg-emerald-500/20 text-emerald-400 border-emerald-500/50"
        sentiment = "BULLISH BIAS"
    elif score <= 45:
        badge_cls = "bg-red-500/20 text-red-400 border-red-500/50"
        sentiment = "BEARISH BIAS"
    else:
        badge_cls = "bg-slate-500/20 text-slate-400 border-slate-500/50"
        sentiment = "NEUTRAL / CHOP"

    # Base Badge
    html = f"""<div class='flex flex-wrap gap-2 mb-3'>
                <span class='px-2 py-1 rounded text-[10px] font-bold border {badge_cls}'>{sentiment}</span>"""

    # 2. Add Context Pills (The "Why")
    if vol == "HIGH":
        html += "<span class='px-2 py-1 rounded text-[10px] font-bold border bg-purple-500/20 text-purple-400 border-purple-500/50'>HIGH VOLATILITY</span>"

    trend = result.trend_score
    if abs(trend) > 50:
        html += "<span class='px-2 py-1 rounded text-[10px] font-bold border bg-blue-500/20 text-blue-400 border-blue-500/50'>STRONG TREND</span>"

    html += "</div>"  # Close badges

    # 3. Narrative Text
    html += f"<p class='leading-relaxed text-slate-300'><strong>Analysis:</strong> The Quant model detects a {direction.lower()} structure with <strong>{score:.1f}% confidence</strong>. "

    if market == "OPTIONS":
        strategy = result.extras.get("opt_strategy", "WAIT")
        theta = result.extras.get("theta_risk", "MODERATE")
        html += f"Implied Volatility conditions ({vol}) favor a <strong>{strategy}</strong> approach. "
        html += f"<span class='block mt-2 text-xs text-slate-500'>* Theta Risk: {theta}</span>"

    elif market == "CRYPTO":
        phase = result.extras.get("market_phase", "NORMAL")
        html += f"Asset is currently in a <strong>{phase}</strong> phase. "
        if "SQUEEZE" in phase:
            html += "Expect explosive expansion soon."
        else:
            html += "Trade the range edges."

    elif market == "FUTURES":
        velocity = result.extras.get("velocity", "NORMAL")
        html += f"Trend Velocity is <strong>{velocity}</strong>. "
        if velocity == "HIGH VELOCITY":
            html += "Momentum is aggressive; avoid fading the move."
        else:
            html += "Price action is grinding; patience required."

    elif market == "EQUITY":
        html += "Price action aligns with Institutional Flows. "
        if vol == "LOW":
            html += "Low volatility supports larger position sizing with tight stops."
        else:
            html += "Defensive sizing recommended due to noise."

    html += "</p>"
    return html


class AnalyzeMarketView(APIView):
    """
    Institutional Grade: Fetches Data -> Runs Quant -> Generates Commentary -> Returns JSON.
    """

    def post(self, request):
        try:
            # 1. Extract Params
            data = request.data
            symbol = data.get("symbol")

            if not symbol:
                return Response({"error": "Symbol is required"}, status=400)

            # 2. FETCH DATA
            symbol_info = {
                "symbol": symbol,
                "token": data.get("token"),
                "exchange": data.get("exchange")
            }

            # Fetch generic underlying data for analysis
            # Options/Futures logic handles the specifics internally
            df = MarketService.get_historical_data(
                symbol_info=symbol_info,
                market_type=data.get("market_type", "EQUITY"),
                trade_style=data.get("trade_style", "SWING")
            )

            if df is None or df.empty:
                # If fail, try fetching index directly for NIFTY/BANKNIFTY options
                if data.get("market_type") == "OPTIONS" and "NIFTY" in symbol:
                    idx_symbol = "NIFTY" if "BANK" not in symbol else "BANKNIFTY"
                    df = MarketService.get_historical_data(
                        {"symbol": idx_symbol}, "FUTURES", data.get("trade_style")
                    )

            if df is None or df.empty:
                return Response({"error": f"Data unavailable for {symbol}. Try a major liquid symbol."}, status=404)

            # 3. RUN QUANT ENGINE
            result = run_quant(
                data.get("market_type", "EQUITY"),
                df,
                symbol,
                data.get("trade_style", "SWING")
            )

            # 4. ENRICH RESPONSE
            response_data = asdict(result)
            response_data["ai_comment"] = generate_quant_commentary(result)

            return Response(response_data)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Engine Error: {str(e)}"}, status=500)


class BacktestView(APIView):
    def post(self, request):
        try:
            data = request.data
            symbol = data.get("symbol")
            market_type = data.get("market_type", "EQUITY")
            trade_style = data.get("trade_style", "SWING")

            symbol_info = {
                "symbol": symbol,
                "token": data.get("token"),
                "exchange": data.get("exchange")
            }

            # Fetch MORE data for backtest (365 days default in service)
            df = MarketService.get_historical_data(
                symbol_info=symbol_info,
                market_type=market_type,
                trade_style=trade_style
            )

            if df is None or len(df) < 50:
                return Response({"error": "Insufficient historical data for backtest (Need 50+ candles)"}, status=400)

            from .quant.quant_engine import get_engine_cls
            engine_cls = get_engine_cls(market_type)

            # Run Event-Driven Backtest
            bt_engine = BacktestEngine(engine_cls, df, symbol, trade_style)
            # Use smaller start_idx if data is short
            warmup = 50 if len(df) > 100 else 20
            results = bt_engine.run(start_idx=warmup)

            return Response(asdict(results))
        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Backtest Failed: {str(e)}"}, status=500)


class MarketUniverseView(APIView):
    def get(self, request):
        markets = get_enabled_markets()
        return Response({"markets": markets})


class SearchSymbolView(APIView):
    def get(self, request):
        q = request.GET.get("q", "")
        market_type = request.GET.get("market_type", "EQUITY")
        results = search_symbols(q, market_type)
        return Response(results)


class SubmitFeedbackView(APIView):
    def post(self, request):
        try:
            data = request.data
            Feedback.objects.create(
                accuracy_rating=data.get("accuracy_rating", 0),
                hallucination_report=data.get("hallucination_report", False),
                comments=data.get("comments", ""),
                symbol=data.get("symbol", ""),
                market=data.get("market", "")
            )
            return Response({"status": "ok"})
        except Exception as e:
            return Response({"error": str(e)}, status=500)