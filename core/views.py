import os
import csv
import hmac
import hashlib
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
from .forms import UserUpdateForm, SignupForm

log = logging.getLogger(__name__)

# --- LEMON SQUEEZY CONFIG ---
LS_API_KEY = os.getenv("LEMONSQUEEZY_API_KEY")
LS_VARIANT_ID = os.getenv("LEMONSQUEEZY_VARIANT_ID")
LS_SIGNING_SECRET = os.getenv("LEMONSQUEEZY_SIGNING_SECRET")


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


# =========================================================
#  PUBLIC & AUTH
# =========================================================

def landing_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    return render(request, 'core/landing.html')


def signup_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # Welcome Email
            try:
                send_mail("Welcome to Reelioo", f"Protocol Initialized for {user.username}.",
                          settings.DEFAULT_FROM_EMAIL, [user.email], fail_silently=True)
            except:
                pass
            return redirect('pricing')
    else:
        form = SignupForm()
    return render(request, 'core/auth/signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == 'POST':
        try:
            from django.contrib.auth.models import User
            user_obj = User.objects.get(email__iexact=request.POST.get('email'))
            user = authenticate(username=user_obj.username, password=request.POST.get('password'))
            if user:
                login(request, user)
                return redirect('terminal')
            else:
                messages.error(request, "Invalid credentials.")
        except:
            messages.error(request, "User not found.")
    return render(request, 'core/auth/login.html')


def logout_view(request):
    logout(request)
    return redirect('landing')


# =========================================================
#  TERMINAL & SETTINGS
# =========================================================

@login_required
def terminal_view(request):
    profile = request.user.profile
    if not profile.is_access_granted():
        messages.warning(request, "Access Expired. Please Renew.")
        return redirect('pricing')
    return render(request, 'core/terminal.html')


@login_required
def settings_view(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return render(request, 'core/auth/profile_saved.html')
    else:
        form = UserUpdateForm(instance=request.user)
    return render(request, 'core/auth/settings.html', {'form': form})


# =========================================================
#  LEMON SQUEEZY PAYMENT
# =========================================================

@login_required
def pricing_view(request):
    url = f"https://reelioo.lemonsqueezy.com/checkout/buy/{LS_VARIANT_ID}?checkout[email]={request.user.email}&checkout[custom][user_id]={request.user.id}"
    return render(request, 'core/pricing.html', {"checkout_url": url, "is_premium": request.user.profile.is_premium})


@login_required
def billing_portal_view(request):
    sub_id = request.user.profile.lemon_squeezy_subscription_id
    if not sub_id: return redirect('pricing')
    try:
        r = requests.get(f"https://api.lemonsqueezy.com/v1/subscriptions/{sub_id}",
                         headers={"Authorization": f"Bearer {LS_API_KEY}", "Accept": "application/vnd.api+json"})
        return redirect(r.json()['data']['attributes']['urls']['customer_portal'])
    except:
        return redirect('settings')


@csrf_exempt
@require_http_methods(["POST"])
def lemon_squeezy_webhook(request):
    signature = request.headers.get("X-Signature")
    if not signature or not LS_SIGNING_SECRET: return HttpResponse("No Sig", status=403)

    digest = hmac.new(LS_SIGNING_SECRET.encode('utf-8'), request.body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, signature): return HttpResponse("Bad Sig", status=403)

    try:
        data = json.loads(request.body)
        meta = data.get('meta', {})
        event = meta.get('event_name')
        attrs = data.get('data', {}).get('attributes', {})

        user_id = attrs.get('checkout_data', {}).get('custom', {}).get('user_id')
        email = attrs.get('user_email')

        from django.contrib.auth.models import User
        user = None
        if user_id:
            try:
                user = User.objects.get(id=user_id)
            except:
                pass
        if not user and email:
            try:
                user = User.objects.get(email__iexact=email)
            except:
                pass

        if user:
            profile = user.profile
            if event in ['subscription_created', 'subscription_updated', 'subscription_resumed']:
                profile.is_premium = True
                profile.subscription_status = attrs.get('status', 'active')
                profile.lemon_squeezy_subscription_id = data.get('data', {}).get('id')
                if attrs.get('renews_at'): profile.subscription_end_date = parser.parse(attrs.get('renews_at'))
                profile.save()
            elif event in ['subscription_cancelled', 'subscription_expired']:
                profile.is_premium = False
                profile.subscription_status = 'expired'
                profile.save()

        return HttpResponse("OK")
    except:
        return HttpResponse("Error", status=500)


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
        res = engine.analyze(df, mode)

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
    """VIP Alpha Scanner"""
    if not has_access(request.user): return HttpResponse("DENIED")

    vip_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    engine = CryptoQuantEngine()
    results = []

    def scan(sym):
        try:
            df = MarketService.get_historical_data(sym, "PERP", "INTRADAY")
            if df.empty: return None
            res = engine.analyze(df, "INTRADAY")
            if res.bias in ["LONG", "SHORT"] and res.score >= 70:
                return {'symbol': sym, 'bias': res.bias, 'score': res.score, 'entry': res.entry,
                        'stop': res.stop, 'target': res.target1, 'explanation': getattr(res, 'narrative', '')}
        except:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for r in executor.map(scan, vip_assets):
            if r: results.append(r)

    if not results: return render(request, 'core/partials/alpha_result.html', {'res': {'found': False}})
    results.sort(key=lambda x: x['score'], reverse=True)
    return render(request, 'core/partials/alpha_result.html', {'res': {'found': True, **results[0]}})


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
    entry = get_object_or_404(JournalEntry, id=entry_id, user=request.user)
    if entry.status == 'PENDING':
        try:
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
            if df is not None:
                curr = float(df['close'].iloc[-1])
                new_status = 'PENDING'

                # Check outcome against Target/Stop
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

                if new_status != 'PENDING':
                    entry.status = new_status
                    # Removed: entry.exit_price = curr (Field doesn't exist)
                    entry.save()
        except:
            pass
    return render(request, 'core/partials/journal_row.html', {'entry': entry})


@login_required
def delete_journal_entry(request, entry_id):
    if request.method == "DELETE":
        JournalEntry.objects.filter(id=entry_id, user=request.user).delete()
        return HttpResponse("")
    return JsonResponse({'status': 'error'}, status=400)


@login_required
def add_journal_entry(request):
    if request.method == "POST":
        return hx_journal_add(request)
    return JsonResponse({'status': 'error'})


@login_required
def ops_dashboard_view(request):
    # 1. Security Check
    if not request.user.is_superuser:
        return redirect('terminal')

    from django.contrib.auth.models import User
    from .models import JournalEntry, UserProfile

    # 2. Safe Metrics Calculation
    users = User.objects.count()
    subs = UserProfile.objects.filter(is_premium=True).count()

    signals = JournalEntry.objects.count()
    wins = JournalEntry.objects.filter(status='WIN').count()

    # Avoid ZeroDivisionError
    win_rate = round((wins / signals * 100), 1) if signals > 0 else 0

    # 3. Crash Prevention: Filter out orphaned trades (user=None)
    # This prevents the "NoneType has no attribute username" error
    feed = JournalEntry.objects.select_related('user') \
               .filter(user__isnull=False) \
               .order_by('-created_at')[:50]

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
    # Auth
    if secret_key and secret_key == getattr(settings, 'CRON_SECRET', 'super-secret-password-123'):
        pass
    elif request.user.is_authenticated and request.user.is_staff:
        pass
    else:
        return JsonResponse({'status': 'forbidden'}, status=403)

    from django.contrib.auth.models import User
    JournalEntry.objects.filter(created_at__lt=timezone.now() - timedelta(days=90)).delete()

    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']
    active_users = User.objects.filter(
        Q(is_superuser=True) | Q(profile__subscription_end_date__gt=timezone.now())).distinct()
    engine = CryptoQuantEngine()
    sent = 0

    for symbol in watchlist:
        try:
            df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
            if df is None: continue
            res = engine.analyze(df, "INTRADAY")

            if res.score >= 65 and res.bias in ['LONG', 'SHORT']:
                admin = User.objects.filter(is_superuser=True).first()
                if admin and not JournalEntry.objects.filter(user=admin, symbol=symbol, status='PENDING',
                                                             created_at__gte=timezone.now() - timedelta(
                                                                     hours=4)).exists():
                    send_discord_alert(symbol, "SNIPER")
                    for user in active_users:
                        if not JournalEntry.objects.filter(user=user, symbol=symbol, status='PENDING').exists():
                            JournalEntry.objects.create(
                                user=user, symbol=symbol, bias=res.bias, entry_price=res.entry,
                                stop_loss=res.stop, target=res.target1, confidence=res.score, status='PENDING'
                            )
                    sent += 1
        except:
            continue

    return JsonResponse({'status': 'success', 'signals': sent})


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


def global_symbols_view(request): return JsonResponse(["BTCUSDT", "ETHUSDT", "SOLUSDT"], safe=False)


def search_crypto_view(request): return JsonResponse(MarketService.search_assets(request.GET.get("q", "")), safe=False)