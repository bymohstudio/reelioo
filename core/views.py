import os
import csv
import json
import logging
import requests
import concurrent.futures
from datetime import datetime
from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.conf import settings

from .quant.crypto_engine import CryptoQuantEngine
from .backtest.backtest_engine import CryptoBacktestEngine
from .services.marketdata_service import MarketService
from .models import JournalEntry

log = logging.getLogger(__name__)


def generate_market_narrative(res):
    vectors = [f.get('desc', '').lower() for f in res.top_features]
    vector_str = " ".join(vectors)
    bias, score = res.bias, res.score

    if score >= 75:
        if bias == "LONG":
            if "volume" in vector_str or "whale" in vector_str:
                return "Aggressive institutional absorption detected at lows. Supply is exhausted."
            elif "breakout" in vector_str:
                return "High-velocity breakout in progress. Order flow is heavily one-sided."
            else:
                return "Market structure has shifted bullish across multiple timeframes. Dips are for buying."
        elif bias == "SHORT":
            if "volume" in vector_str:
                return "Heavy distribution spotted. Smart money is offloading into retail strength."
            else:
                return "Technical structure has broken down. Expect lower prices as stops get triggered."
    elif score >= 50:
        return "Momentum is building, but conviction is moderate. Wait for confirmation."

    return "Liquidity is thin and direction is unclear. Expect stop-hunts."


def landing_view(request):
    return render(request, 'core/landing.html')


def terminal_view(request):
    return render(request, 'core/terminal.html')


def pricing_view(request):
    return render(request, 'core/pricing.html', {"is_premium": False})


@require_http_methods(["GET"])
def hx_ticker(request):
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    data = []
    for s in symbols:
        try:
            df = MarketService.get_historical_data(f"{s}USDT", "PERP", "SCALP")
            if df is not None and not df.empty:
                price = float(df['close'].iloc[-1])
                prev = float(df['close'].iloc[-2]) if len(df) > 1 else price
                direction = 'up' if price > prev else ('down' if price < prev else 'flat')
                fmt = f"${price:,.2f}" if price > 1.0 else f"${price:,.4f}"
                data.append({'symbol': s, 'price': fmt, 'direction': direction})
            else:
                data.append({'symbol': s, 'price': "---", 'direction': 'flat'})
        except:
            data.append({'symbol': s, 'price': "---", 'direction': 'flat'})
    return render(request, 'core/partials/ticker.html', {'ticker': data * 4})


@require_http_methods(["POST"])
def hx_analyze(request):
    symbol = request.POST.get("symbol", "").upper().strip()
    if not symbol.endswith("USDT"): symbol += "USDT"
    mode = request.POST.get("mode", "INTRADAY")

    try:
        df = MarketService.get_historical_data(symbol, "PERP", mode)
        if df is None or df.empty: return HttpResponse('<div class="text-red-500">DATA ERROR</div>')

        engine = CryptoQuantEngine()
        res = engine.analyze(df, mode, symbol=symbol)

        note_tag = "TRENDING" if res.score >= 60 else "CHOPPY"
        note_msg = generate_market_narrative(res)

        return render(request, 'core/partials/dashboard.html', {
            'res': res, 'symbol': symbol, 'mode': mode, 'note_tag': note_tag, 'note_msg': note_msg
        })
    except Exception as e:
        return HttpResponse(f'<div class="text-red-500">ERROR: {e}</div>')


@require_http_methods(["POST"])
def hx_backtest(request):
    try:
        symbol = request.POST.get("symbol", "BTCUSDT")
        df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
        engine = CryptoBacktestEngine(df, symbol)
        stats = engine.run("INTRADAY")
        return render(request, 'core/partials/backtest_result.html', {'stats': stats})
    except Exception as e:
        return HttpResponse(f"Error: {e}")


@require_http_methods(["GET"])
def hx_alpha_scan(request):
    all_assets = MarketService._load_exchange_info()
    scan_list = [a['symbol'] for a in all_assets[:30]]

    engine = CryptoQuantEngine()
    results = []

    def scan(sym):
        try:
            scan_mode = "INTRADAY"
            df = MarketService.get_historical_data(sym, "PERP", scan_mode)

            if df is None or df.empty or len(df) < 210:
                return None

            res = engine.analyze(df, trade_style=scan_mode, symbol=sym)

            if res.bias not in ["LONG", "SHORT"] or res.score < 75:
                return None

            df_prev = df.iloc[:-1]
            res_prev = engine.analyze(df_prev, trade_style=scan_mode, symbol=sym)

            if res_prev.bias != res.bias:
                return None

            return {
                'symbol': sym,
                'bias': res.bias,
                'score': res.score,
                'lane': res.lane,
                'entry': res.entry,
                'stop': res.stop,
                'target': res.target,
                'explanation': getattr(res, 'narrative', 'Setup Detected'),
                'timestamp': str(df.index[-1])
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(scan, scan_list):
            if r:
                results.append(r)

    if not results:
        return render(request, 'core/partials/alpha_result.html', {'res': {'found': False}})

    results.sort(key=lambda x: x['score'], reverse=True)
    best_setup = results[0]

    return render(request, 'core/partials/alpha_result.html', {'res': {'found': True, **best_setup}})


@require_http_methods(["POST"])
def hx_journal_add(request):
    if not request.user.is_authenticated:
        return HttpResponse('<button class="w-full mt-8 bg-red-500/10 text-red-500 p-4 rounded-xl">LOGIN REQUIRED</button>')
    try:
        JournalEntry.objects.create(
            user=request.user, symbol=request.POST.get('symbol'),
            bias=request.POST.get('bias'), entry_price=float(request.POST.get('entry', 0)),
            stop_loss=float(request.POST.get('stop', 0)), target=float(request.POST.get('target', 0)),
            confidence=float(request.POST.get('score', 0)), status='PENDING'
        )
        return HttpResponse(
            '<button class="w-full mt-8 bg-emerald-500/10 text-emerald-400 p-4 rounded-xl">SAVED TO JOURNAL</button>')
    except:
        return HttpResponse('<button class="w-full mt-8 bg-red-500/10 text-red-500 p-4 rounded-xl">ERROR</button>')


def journal_view(request):
    if not request.user.is_authenticated:
        return redirect('terminal')

    entries_list = JournalEntry.objects.filter(user=request.user).order_by('-created_at')

    net_roi = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in entries_list:
        if trade.status in ['WIN', 'LOSS'] and trade.entry_price > 0:
            exit_price = trade.target if trade.status == 'WIN' else trade.stop_loss

            if trade.bias == 'LONG':
                pct = (exit_price - trade.entry_price) / trade.entry_price
            else:
                pct = (trade.entry_price - exit_price) / trade.entry_price

            pnl = pct * 100
            net_roi += pnl
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else round(gross_profit, 2)

    paginator = Paginator(entries_list, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    return render(request, 'core/journal.html', {
        'page_obj': page_obj, 'net_roi': round(net_roi, 2),
        'profit_factor': profit_factor, 'active_pending': entries_list.filter(status='PENDING').count()
    })


def refresh_journal_entry(request, entry_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error'}, status=403)

    from .services.discord_service import send_win_prompt

    entry = get_object_or_404(JournalEntry, id=entry_id, user=request.user)

    msg_type = "info"
    title = "Synced"
    message = "Trade is already closed."

    if entry.status == 'PENDING':
        ttl_hours = 4.0
        hours_elapsed = (timezone.now() - entry.created_at).total_seconds() / 3600

        if hours_elapsed >= ttl_hours:
            final_status = 'EXPIRED'
            try:
                df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
                if df is not None and not df.empty:
                    trade_start = entry.created_at
                    recent = df[df.index >= trade_start.strftime('%Y-%m-%d %H:%M:%S')] if hasattr(df.index, 'strftime') else df.tail(48)

                    if len(recent) > 0:
                        period_high = float(recent['high'].max())
                        period_low = float(recent['low'].min())

                        if entry.bias == 'LONG':
                            if period_high >= entry.target:
                                final_status = 'WIN'
                            elif period_low <= entry.stop_loss:
                                for _, candle in recent.iterrows():
                                    if float(candle['low']) <= entry.stop_loss:
                                        final_status = 'LOSS'
                                        break
                                    if float(candle['high']) >= entry.target:
                                        final_status = 'WIN'
                                        break
                        else:
                            if period_low <= entry.target:
                                final_status = 'WIN'
                            elif period_high >= entry.stop_loss:
                                for _, candle in recent.iterrows():
                                    if float(candle['high']) >= entry.stop_loss:
                                        final_status = 'LOSS'
                                        break
                                    if float(candle['low']) <= entry.target:
                                        final_status = 'WIN'
                                        break
            except Exception:
                pass

            entry.status = final_status
            entry.save()

            if final_status == 'WIN':
                msg_type = "success"
                title = "Target Hit (Retroactive)"
                message = "Target was reached during the signal window."

                if request.user.is_superuser:
                    try:
                        roi = abs((entry.target - entry.entry_price) / entry.entry_price) * 100
                        send_win_prompt(entry.symbol, round(roi, 2), hours_elapsed)
                    except Exception as e:
                        print(f"Marketing Error: {e}")

            elif final_status == 'LOSS':
                msg_type = "error"
                title = "Stop Hit"
                message = "Stop loss was reached during the signal window."
            else:
                msg_type = "info"
                title = "Signal Expired"
                message = "4h window closed. Neither target nor stop was hit."

        else:
            try:
                df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
                if df is not None and not df.empty:
                    curr = float(df['close'].iloc[-1])
                    message = f"Current Price: ${curr}"
                    new_status = 'PENDING'

                    trade_time = entry.created_at
                    try:
                        if hasattr(df.index, 'tz_localize'):
                            trade_ts = pd.Timestamp(trade_time).tz_localize(None)
                        else:
                            trade_ts = pd.Timestamp(trade_time)
                        recent = df[df.index >= trade_ts]
                    except Exception:
                        recent = df.tail(2)

                    if len(recent) == 0:
                        if entry.bias == 'LONG':
                            if curr >= entry.target: new_status = 'WIN'
                            elif curr <= entry.stop_loss: new_status = 'LOSS'
                        else:
                            if curr <= entry.target: new_status = 'WIN'
                            elif curr >= entry.stop_loss: new_status = 'LOSS'
                    else:
                        period_high = float(recent['high'].max())
                        period_low = float(recent['low'].min())

                        if entry.bias == 'LONG':
                            if period_high >= entry.target: new_status = 'WIN'
                            elif period_low <= entry.stop_loss: new_status = 'LOSS'
                        else:
                            if period_low <= entry.target: new_status = 'WIN'
                            elif period_high >= entry.stop_loss: new_status = 'LOSS'

                    if new_status != 'PENDING':
                        entry.status = new_status
                        entry.save()

                        if new_status == 'WIN':
                            msg_type = "success"
                            title = "Target Hit!"
                            message = "Trade closed in profit."

                            if request.user.is_superuser:
                                try:
                                    duration = (timezone.now() - entry.created_at).total_seconds() / 3600
                                    roi = abs((entry.target - entry.entry_price) / entry.entry_price) * 100
                                    send_win_prompt(entry.symbol, round(roi, 2), duration)
                                except Exception as e:
                                    print(f"Marketing Error: {e}")

                        elif new_status == 'LOSS':
                            msg_type = "error"
                            title = "Stop Loss Hit"
                            message = "Trade closed in loss."
                    else:
                        title = "Status: PENDING"

            except Exception:
                msg_type = "warning"
                title = "Sync Error"
                message = "Could not fetch market data."

    else:
        if entry.status == 'WIN':
            msg_type = "success"
            title = "Trade Complete"
            message = "Result: WIN (Alert Sent)"

            if request.user.is_superuser:
                try:
                    duration = (timezone.now() - entry.created_at).total_seconds() / 3600
                    roi = abs((entry.target - entry.entry_price) / entry.entry_price) * 100
                    send_win_prompt(entry.symbol, round(roi, 2), duration)
                except Exception as e:
                    print(f"Marketing Error: {e}")

        elif entry.status == 'LOSS':
            msg_type = "error"
            title = "Trade Complete"
            message = "Result: LOSS"

        elif entry.status in ['EXPIRED', 'INVALID']:
            msg_type = "info"
            title = "Signal Expired"
            message = "Trade window closed."

    response = render(request, 'core/partials/journal_row.html', {'entry': entry})
    response['HX-Trigger'] = json.dumps({'showToast': {'type': msg_type, 'title': title, 'message': message}})
    return response


def delete_journal_entry(request, entry_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error'}, status=403)

    if request.method == "DELETE":
        JournalEntry.objects.filter(id=entry_id, user=request.user).delete()
        response = HttpResponse("")
        response['HX-Trigger'] = json.dumps({'showToast': {'type': 'success', 'title': 'Deleted', 'message': 'Journal entry removed.'}})
        return response

    return JsonResponse({'status': 'error'}, status=400)


def add_journal_entry(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error'}, status=403)
    if request.method == "POST":
        return hx_journal_add(request)
    return JsonResponse({'status': 'error'})


def ops_dashboard_view(request):
    if not request.user.is_superuser:
        return redirect('terminal')

    from django.contrib.auth.models import User
    from .models import JournalEntry, UserProfile

    users = User.objects.count()
    subs = UserProfile.objects.filter(is_premium=True).count()
    signals = JournalEntry.objects.count()
    wins = JournalEntry.objects.filter(status='WIN').count()
    win_rate = round((wins / signals * 100), 1) if signals > 0 else 0
    feed = JournalEntry.objects.select_related('user').filter(user__isnull=False).order_by('-created_at')[:50]

    return render(request, 'core/ops_dashboard.html', {
        'total_users': users, 'active_subs': subs,
        'total_signals': signals, 'win_rate': win_rate, 'recent_signals': feed
    })


def send_discord_alert(symbol, alert_type="SNIPER"):
    webhook_url = os.getenv('DISCORD_URL')
    if not webhook_url: return

    terminal_link = f"https://reelioo.app/terminal?ticker={symbol}"

    if alert_type == "SNIPER":
        color = 5814783
        title = f"🎯 SNIPER TARGET IDENTIFIED: {symbol}"
        description = (
            "**Institutional Activity Detected.**\n"
            "Neural engines have locked onto a high-probability setup.\n\n"
            "Analyzing Order Flow, Whale Volume, and Trend Vectors..."
        )
        thumbnail = "https://cdn-icons-png.flaticon.com/512/3121/3121575.png"
    else:
        color = 16776960
        title = f"📡 RADAR CONTACT: {symbol}"
        description = (
            "**Volatility Spike Detected.**\n"
            "Abnormal market behavior observed. Risk protocols active."
        )
        thumbnail = "https://cdn-icons-png.flaticon.com/512/564/564619.png"

    payload = {
        "username": "Reelioo Intelligence",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "thumbnail": {"url": thumbnail},
            "fields": [
                {"name": "Asset", "value": f"`{symbol}`", "inline": True},
                {"name": "Signal Strength", "value": "██████▒▒▒▒ **[HIDDEN]**", "inline": True},
                {"name": "Full Analysis", "value": f"👉 [**CLICK TO REVEAL DATA**]({terminal_link})", "inline": False}
            ],
            "footer": {
                "text": "🔒 Reelioo Terminal",
                "icon_url": "https://cdn-icons-png.flaticon.com/512/2913/2913133.png"
            },
            "timestamp": datetime.utcnow().isoformat()
        }]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"Discord Error: {e}")


def cron_scan_trigger(request, secret_key=None):
    authorized = False
    if secret_key and secret_key == getattr(settings, 'CRON_SECRET', 'super-secret-password-123'):
        authorized = True
    elif request.user.is_authenticated and request.user.is_superuser:
        authorized = True

    if not authorized:
        return JsonResponse({'status': 'forbidden'}, status=403)

    from django.contrib.auth.models import User
    from django.db.models import Q
    from django.utils import timezone
    from datetime import timedelta
    from .models import JournalEntry
    from .quant.crypto_engine import CryptoQuantEngine
    from .services.marketdata_service import MarketService
    from .services.discord_service import send_marketing_prompt

    try:
        JournalEntry.objects.filter(created_at__lt=timezone.now() - timedelta(days=90)).delete()
    except:
        pass

    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']

    active_users = User.objects.filter(
        Q(is_superuser=True) | Q(profile__is_premium=True)
    ).distinct()

    engine = CryptoQuantEngine()
    sent = 0

    for symbol in watchlist:
        try:
            df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
            if df is None: continue

            res = engine.analyze(df, "INTRADAY")

            if res.score >= 65 and res.bias in ['LONG', 'SHORT']:
                admin = User.objects.filter(is_superuser=True).first()
                exists = False

                if admin:
                    exists = JournalEntry.objects.filter(
                        user=admin, symbol=symbol, status='PENDING',
                        created_at__gte=timezone.now() - timedelta(hours=4)
                    ).exists()

                if not exists:
                    send_discord_alert(symbol, "SNIPER")

                    if res.score >= 75:
                        try:
                            whale_state = getattr(res, 'whale_state', 'BASELINE')
                            send_marketing_prompt(symbol, res.score, whale_state)
                        except Exception as e:
                            print(f"Marketing Trigger Error: {e}")

                    for user in active_users:
                        if not JournalEntry.objects.filter(user=user, symbol=symbol, status='PENDING').exists():
                            JournalEntry.objects.create(
                                user=user, symbol=symbol, bias=res.bias,
                                entry_price=res.entry, stop_loss=res.stop,
                                target=res.target, confidence=res.score, status='PENDING'
                            )
                    sent += 1
        except Exception as e:
            print(f"Scan Error {symbol}: {e}")
            continue

    return JsonResponse({'status': 'success', 'signals': sent})


def global_symbols_view(request):
    csv_path = os.path.join(settings.BASE_DIR, 'global_symbols.csv')
    symbols = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'symbol' in row: symbols.append(row['symbol'])
        except:
            pass

    if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    return JsonResponse(symbols, safe=False)


def search_crypto_view(request):
    return JsonResponse(MarketService.search_assets(request.GET.get("q", "")), safe=False)


def terms_view(request): return render(request, 'core/legal/terms.html')
def about_view(request): return render(request, 'core/why_reelioo.html')
def privacy_view(request): return render(request, 'core/legal/privacy.html')
def refund_view(request): return render(request, 'core/legal/refund.html')
def contact_view(request): return render(request, 'core/legal/contact.html')
def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')
def robots_view(request): return HttpResponse("User-agent: *\nDisallow:", content_type="text/plain")
def sitemap_view(request): return HttpResponse("", content_type="application/xml")
