import os
import csv
import json
import logging
import requests
import concurrent.futures
from dateutil import parser
from datetime import datetime, timedelta

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.html import strip_tags
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.mail import send_mail
from django.conf import settings

from .quant.crypto_engine import CryptoQuantEngine
from .backtest.backtest_engine import CryptoBacktestEngine
from .services.marketdata_service import MarketService
from .models import JournalEntry, UserProfile
from .services.gumroad_service import verify_gumroad_license

log = logging.getLogger(__name__)


# =========================================================
#  HELPERS
# =========================================================

def has_access(user):
    """Checks if user has premium access via Profile Property"""
    return user.profile.is_access_granted()


def generate_market_narrative(res):
    """Generates 'Trader Speak' based on math vectors."""
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


def verify_subscription_status(user):
    """
    Helper to check Gumroad status periodically.
    Used in terminal_view and get_access_view.
    """
    if not hasattr(user, 'profile'): return

    profile = user.profile

    # Skip check for superusers
    if user.is_superuser: return

    # Only check if 24 hours passed since last check
    if profile.needs_verification():
        is_valid, msg = verify_gumroad_license(profile.gumroad_license_key)

        if is_valid:
            profile.is_premium = True
            profile.last_verified_at = timezone.now()
        else:
            profile.is_premium = False  # Revoke access

        profile.save()


# =========================================================
#  AUTH (UNIFIED GUMROAD FLOW)
# =========================================================

def get_access_view(request):
    """
    The Single Entry Point.
    User enters Username + Key.
    - If user exists -> Login.
    - If new -> Create + Login.
    """
    if request.user.is_authenticated:
        return redirect('terminal')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        key = request.POST.get('license_key', '').strip()

        if not username or not key:
            messages.error(request, "Username and License Key are required.")
            return render(request, 'core/auth/get_access.html')

        # This triggers LicenseKeyBackend.authenticate
        user = authenticate(request, username=username, license_key=key)

        if user:
            login(request, user)
            # Check status immediately on fresh login
            verify_subscription_status(user)
            return redirect('terminal')
        else:
            messages.error(request, "Access Denied. Invalid Key or Username mismatch.")

    return render(request, 'core/auth/get_access.html')


def logout_view(request):
    logout(request)
    return redirect('landing')


def landing_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    return render(request, 'core/landing.html')


# =========================================================
#  TERMINAL
# =========================================================

@login_required
def terminal_view(request):
    # 1. Periodically verify license (e.g. daily)
    verify_subscription_status(request.user)

    profile = request.user.profile
    if not profile.is_access_granted():
        messages.warning(request, "Access Expired. Please Renew.")
        return redirect('pricing')

    return render(request, 'core/terminal.html')


# =========================================================
#  PAYMENT (PRICING ONLY)
# =========================================================

@login_required
def pricing_view(request):
    # Just render the template with the Gumroad link
    # No API logic needed here anymore
    return render(request, 'core/pricing.html', {"is_premium": request.user.profile.is_premium})


# =========================================================
#  HTMX PARTIALS
# =========================================================

@require_http_methods(["GET"])
def hx_ticker(request):
    """Top bar live price ticker"""
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


@login_required
@require_http_methods(["POST"])
def hx_analyze(request):
    """Main Terminal Analysis View"""
    if not has_access(request.user): return HttpResponse('<div class="text-red-500">ACCESS DENIED</div>')

    symbol = request.POST.get("symbol", "").upper().strip()
    if not symbol.endswith("USDT"): symbol += "USDT"
    mode = request.POST.get("mode", "INTRADAY")

    try:
        df = MarketService.get_historical_data(symbol, "PERP", mode)
        if df is None or df.empty: return HttpResponse('<div class="text-red-500">DATA ERROR</div>')

        engine = CryptoQuantEngine()

        # [CRITICAL UPDATE] You MUST pass 'symbol=symbol' here!
        # Without this, the engine cannot fetch the Order Book / Order Flow.
        res = engine.analyze(df, mode, symbol=symbol)

        note_tag = "TRENDING" if res.score >= 60 else "CHOPPY"
        note_msg = generate_market_narrative(res)

        return render(request, 'core/partials/dashboard.html', {
            'res': res, 'symbol': symbol, 'mode': mode, 'note_tag': note_tag, 'note_msg': note_msg
        })
    except Exception as e:
        return HttpResponse(f'<div class="text-red-500">ERROR: {e}</div>')


@login_required
@require_http_methods(["POST"])
def hx_backtest(request):
    """Backtest execution"""
    try:
        symbol = request.POST.get("symbol", "BTCUSDT")
        df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
        engine = CryptoBacktestEngine(df, symbol)
        stats = engine.run("INTRADAY")
        return render(request, 'core/partials/backtest_result.html', {'stats': stats})
    except Exception as e:
        return HttpResponse(f"Error: {e}")


@login_required
@require_http_methods(["GET"])
def hx_alpha_scan(request):
    """
    THE HUNTER: Automatically scans the Top 30 Liquid Assets.
    Finds opportunities even when BTC is dead.
    """
    if not has_access(request.user): return HttpResponse("DENIED")

    # 1. DYNAMIC ASSET LOADING (No more hardcoded lists)
    # We fetch the top 30 coins by Volume from your MarketService
    all_assets = MarketService._load_exchange_info()

    # Take Top 30 Liquid Assets (Skip stablecoins if not filtered in Service)
    scan_list = [a['symbol'] for a in all_assets[:30]]

    engine = CryptoQuantEngine()
    results = []

    def scan(sym):
        try:
            # FORCE 'SCALP' MODE
            # We use SCALP because it catches the pumps (KAIA, SUI) that SWING misses
            df = MarketService.get_historical_data(sym, "PERP", "SCALP")

            if df is None or df.empty: return None

            # Analyze using v34 Dual Core
            res = engine.analyze(df, trade_style="SCALP", symbol=sym)

            # Filter: Only show distinct "Long/Short" signals (Ignore Watch/Hold)
            if res.bias in ["LONG", "SHORT"] and res.score >= 70:
                return {
                    'symbol': sym,
                    'bias': res.bias,
                    'score': res.score,
                    'lane': res.lane,  # Show "SCALP ENTRY" or "TREND"
                    'entry': res.entry,
                    'stop': res.stop,
                    'target': res.target1,
                    'explanation': getattr(res, 'narrative', 'Setup Detected')
                }
        except:
            return None

    # 2. PARALLEL EXECUTION (Speed is key for UX)
    # Scans 30 coins in ~2 seconds using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(scan, scan_list):
            if r: results.append(r)

    # 3. RANKING
    # Sort by Confidence Score (Highest first)
    if not results:
        return render(request, 'core/partials/alpha_result.html', {'res': {'found': False}})

    results.sort(key=lambda x: x['score'], reverse=True)

    # Return the Single Best Setup (or loop in template to show Top 3)
    best_setup = results[0]
    return render(request, 'core/partials/alpha_result.html', {'res': {'found': True, **best_setup}})


@login_required
@require_http_methods(["POST"])
def hx_journal_add(request):
    """Adds manual trade from Terminal"""
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


# =========================================================
#  JOURNAL & OPS
# =========================================================

@login_required
def journal_view(request):
    profile = request.user.profile
    if not profile.is_access_granted(): return redirect('pricing')

    entries_list = JournalEntry.objects.filter(user=request.user).order_by('-created_at')

    # Metrics: Percentage Based
    net_roi = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in entries_list:
        if trade.status in ['WIN', 'LOSS'] and trade.entry_price > 0:
            # FIX: Use Target/Stop for calc since 'exit_price' doesn't exist in DB
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


@login_required
def refresh_journal_entry(request, entry_id):
    from .models import JournalEntry
    # Import Discord Service
    from .services.discord_service import send_win_prompt

    entry = get_object_or_404(JournalEntry, id=entry_id, user=request.user)

    # Default vars
    msg_type = "info"
    title = "Synced"
    message = "Trade is already closed."

    # --- CASE 1: TRADE IS ACTIVE (CHECK PRICE & TIME) ---
    if entry.status == 'PENDING':

        # [NEW LOGIC] CHECK TIME-TO-LIVE (4 HOURS)
        hours_elapsed = (timezone.now() - entry.created_at).total_seconds() / 3600

        if hours_elapsed >= 4.0:
            # CHANGE: Mark as 'EXPIRED' instead of 'INVALID'
            entry.status = 'EXPIRED'
            entry.save()

            msg_type = "info"  # Neutral Blue/Gray
            title = "Signal Expired"
            message = "4h time limit reached. Setup closed flat."

        else:
            # Proceed with Price Check since time is valid
            try:
                df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
                if df is not None:
                    curr = float(df['close'].iloc[-1])
                    message = f"Current Price: ${curr}"
                    new_status = 'PENDING'

                    # Check Outcome
                    if entry.bias == 'LONG':
                        if curr >= entry.target:
                            new_status = 'WIN'
                        elif curr <= entry.stop_loss:
                            new_status = 'LOSS'
                    else:
                        if curr <= entry.target:
                            new_status = 'WIN'
                        elif curr >= entry.stop_loss:
                            new_status = 'LOSS'

                    # Update DB if status changed
                    if new_status != 'PENDING':
                        entry.status = new_status
                        entry.save()

                        if new_status == 'WIN':
                            msg_type = "success"
                            title = "Target Hit!"
                            message = "Trade closed in profit."

                            # --- TRIGGER ALERT (TRANSITION) ---
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

            except Exception as e:
                msg_type = "warning"
                title = "Sync Error"
                message = "Could not fetch market data."

    # --- CASE 2: TRADE IS ALREADY CLOSED ---
    else:
        if entry.status == 'WIN':
            msg_type = "success"
            title = "Trade Complete"
            message = "Result: WIN (Alert Sent)"

            # --- ALLOW RE-SENDING ALERT FOR EXISTING WINS ---
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

        elif entry.status == 'EXPIRED':  # Handle the new status here
            msg_type = "info"
            title = "Signal Expired"
            message = "Trade window closed (4h limit)."

        elif entry.status == 'INVALID':  # Keep for backward compatibility
            msg_type = "info"
            title = "Signal Expired"
            message = "Trade window closed."

    # Render response
    response = render(request, 'core/partials/journal_row.html', {'entry': entry})
    response['HX-Trigger'] = json.dumps({
        'showToast': {'type': msg_type, 'title': title, 'message': message}
    })
    return response


@login_required
def delete_journal_entry(request, entry_id):
    from .models import JournalEntry
    if request.method == "DELETE":
        JournalEntry.objects.filter(id=entry_id, user=request.user).delete()

        response = HttpResponse("")
        # Trigger Success Alert
        response['HX-Trigger'] = json.dumps({
            'showToast': {
                'type': 'success',
                'title': 'Deleted',
                'message': 'Journal entry removed.'
            }
        })
        return response

    return JsonResponse({'status': 'error'}, status=400)


@login_required
def add_journal_entry(request):
    if request.method == "POST":
        return hx_journal_add(request)
    return JsonResponse({'status': 'error'})


@login_required
def ops_dashboard_view(request):
    # 1. Access Control: Strict Superuser Only
    if not request.user.is_superuser:
        return redirect('terminal')

    # Import inside to avoid circular import errors
    from django.contrib.auth.models import User
    from .models import JournalEntry, UserProfile

    # 2. METRICS
    users = User.objects.count()

    # FIX: Count users where is_premium is True (Correct for Gumroad)
    subs = UserProfile.objects.filter(is_premium=True).count()

    signals = JournalEntry.objects.count()
    wins = JournalEntry.objects.filter(status='WIN').count()

    # Avoid DivisionByZero if no signals exist
    win_rate = round((wins / signals * 100), 1) if signals > 0 else 0

    # 3. FEED: Filter out broken records (where user is None) to prevent 500 errors
    feed = JournalEntry.objects.select_related('user').filter(user__isnull=False).order_by('-created_at')[:50]

    return render(request, 'core/ops_dashboard.html', {
        'total_users': users,
        'active_subs': subs,
        'total_signals': signals,
        'win_rate': win_rate,
        'recent_signals': feed
    })


# =========================================================
#  CRON & ALERTS
# =========================================================

def send_discord_alert(symbol, alert_type="SNIPER"):
    webhook_url = os.getenv('DISCORD_URL')
    if not webhook_url: return

    # --- 1. GENERATE DEEP LINK ---
    # This link opens the terminal and auto-scans the specific coin
    terminal_link = f"https://reelioo.app/terminal?ticker={symbol}"

    # --- 2. DESIGN THE TEASER ---
    # We use neutral colors and generic terms to avoid leaking 'Long' vs 'Short'

    if alert_type == "SNIPER":
        # Purple/Blue gradient feel (Hex: 5865F2 - Discord Blurple)
        # We don't use Green/Red here so they don't guess the direction.
        color = 5814783
        title = f"🎯 SNIPER TARGET IDENTIFIED: {symbol}"
        description = (
            "**Institutional Activity Detected.**\n"
            "Neural engines have locked onto a high-probability setup.\n\n"
            "Analyzing Order Flow, Whale Volume, and Trend Vectors..."
        )
        thumbnail = "https://cdn-icons-png.flaticon.com/512/3121/3121575.png"  # Target Icon

    else:
        # Orange/Yellow for Warning
        color = 16776960
        title = f"📡 RADAR CONTACT: {symbol}"
        description = (
            "**Volatility Spike Detected.**\n"
            "Abnormal market behavior observed. Risk protocols active."
        )
        thumbnail = "https://cdn-icons-png.flaticon.com/512/564/564619.png"  # Radar Icon

    # --- 3. CONSTRUCT PAYLOAD ---
    payload = {
        "username": "Reelioo Intelligence",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "thumbnail": {"url": thumbnail},
            "fields": [
                {
                    "name": "Asset",
                    "value": f"`{symbol}`",
                    "inline": True
                },
                {
                    "name": "Signal Strength",
                    "value": "██████▒▒▒▒ **[HIDDEN]**",  # Visual bar to tease
                    "inline": True
                },
                {
                    "name": "Full Analysis",
                    "value": f"👉 [**CLICK TO REVEAL DATA**]({terminal_link})",
                    "inline": False
                }
            ],
            "footer": {
                "text": "🔒 Auth Required • Reelioo Terminal",
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
    # 1. Auth Check
    authorized = False
    if secret_key and secret_key == getattr(settings, 'CRON_SECRET', 'super-secret-password-123'):
        authorized = True
    elif request.user.is_authenticated and request.user.is_superuser:
        authorized = True

    if not authorized:
        return JsonResponse({'status': 'forbidden'}, status=403)

    # 2. Imports
    from django.contrib.auth.models import User
    from django.db.models import Q
    from django.utils import timezone
    from datetime import timedelta
    from .models import JournalEntry
    from .quant.crypto_engine import CryptoQuantEngine
    from .services.marketdata_service import MarketService
    # REPLACED: TwitterBot with Discord Marketing Prompt
    from .services.discord_service import send_marketing_prompt

    # 3. Cleanup
    try:
        JournalEntry.objects.filter(created_at__lt=timezone.now() - timedelta(days=90)).delete()
    except:
        pass

    # 4. Watchlist
    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']

    active_users = User.objects.filter(
        Q(is_superuser=True) |
        Q(profile__is_premium=True)  # CHANGED: Now uses is_premium for Gumroad
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
                        user=admin,
                        symbol=symbol,
                        status='PENDING',
                        created_at__gte=timezone.now() - timedelta(hours=4)
                    ).exists()

                if not exists:
                    # A. Public Alert (Uses your existing function in views.py)
                    send_discord_alert(symbol, "SNIPER")

                    # B. Marketing Alert (Click-to-Tweet for Admin)
                    if res.score >= 75:
                        try:
                            whale_state = getattr(res, 'whale_state', 'BASELINE')
                            send_marketing_prompt(symbol, res.score, whale_state)
                        except Exception as e:
                            print(f"Marketing Trigger Error: {e}")

                    # C. Distribute to Users
                    for user in active_users:
                        if not JournalEntry.objects.filter(user=user, symbol=symbol, status='PENDING').exists():
                            JournalEntry.objects.create(
                                user=user,
                                symbol=symbol,
                                bias=res.bias,
                                entry_price=res.entry,
                                stop_loss=res.stop,
                                target=res.target1,
                                confidence=res.score,
                                status='PENDING'
                            )
                    sent += 1
        except Exception as e:
            print(f"Scan Error {symbol}: {e}")
            continue

    return JsonResponse({'status': 'success', 'signals': sent})


# =========================================================
#  STATIC & SEARCH (CSV LOGIC RESTORED)
# =========================================================

@login_required
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

    # Fallback if CSV empty or failed
    if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    return JsonResponse(symbols, safe=False)


def search_crypto_view(request):
    return JsonResponse(MarketService.search_assets(request.GET.get("q", "")), safe=False)


# =========================================================
#  STATIC
# =========================================================
def terms_view(request): return render(request, 'core/legal/terms.html')


def about_view(request): return render(request, 'core/why_reelioo.html')


def privacy_view(request): return render(request, 'core/legal/privacy.html')


def refund_view(request): return render(request, 'core/legal/refund.html')


def contact_view(request): return render(request, 'core/legal/contact.html')


def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


def robots_view(request): return HttpResponse("User-agent: *\nDisallow:", content_type="text/plain")


def sitemap_view(request): return HttpResponse("", content_type="application/xml")