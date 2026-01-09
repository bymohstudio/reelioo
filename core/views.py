import os
import csv
import hmac
import hashlib
import json
import logging
import random  # Added for narrative variety
import requests
import concurrent.futures
from dateutil import parser
from datetime import timedelta
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator
from django.contrib.auth.models import User
from django.db.models import Q
from django.conf import settings
from django.contrib import messages

from .quant.crypto_engine import CryptoQuantEngine
from .backtest.backtest_engine import CryptoBacktestEngine
from .services.marketdata_service import MarketService
from .models import JournalEntry, UserProfile
from .forms import UserUpdateForm, SignupForm

log = logging.getLogger(__name__)

# --- CONFIG ---
LS_API_KEY = os.getenv("LEMONSQUEEZY_API_KEY")
LS_VARIANT_ID = os.getenv("LEMONSQUEEZY_VARIANT_ID")
LS_SIGNING_SECRET = os.getenv("LEMONSQUEEZY_SIGNING_SECRET")


# --- HELPERS ---

def has_access(user):
    """Checks if user has premium access via Profile Property"""
    return user.profile.is_premium


def send_discord_alert(symbol, data=None, alert_type="SNIPER"):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url: return

    terminal_link = f"https://reelioo.app/terminal?ticker={symbol}"

    if alert_type == "SNIPER":
        color = 5814783
        title = f"🎯 SNIPER TARGET: {symbol}"
        description = "**Institutional Activity Detected.**\nQuant engine locked onto high-probability setup."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/3121/3121575.png"
    elif alert_type == "WIN":
        color = 5763719
        title = f"✅ TARGET HIT: {symbol}"
        description = "Trade closed in profit."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/190/190411.png"
    elif alert_type == "LOSS":
        color = 15548997
        title = f"🛑 STOP HIT: {symbol}"
        description = "Trade closed at stop loss."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/1828/1828843.png"
    else:
        color = 16776960
        title = f"📡 RADAR CONTACT: {symbol}"
        description = "Volatility Spike Detected."
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
                {"name": "Action", "value": "See Terminal for Logic Vectors", "inline": True},
                {"name": "Full Analysis", "value": f"👉 [**OPEN TERMINAL**]({terminal_link})", "inline": False}
            ],
            "footer": {"text": "🔒 Auth Required • Reelioo Terminal"},
            "timestamp": timezone.now().isoformat()
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=3)
    except:
        pass


def generate_market_narrative(res):
    """
    Deterministically generates 'Trader Speak' based on math vectors.
    Ensures the note adds CONTEXT, not just repeats the list.
    """
    # 1. Extract Logic Signals (lowercase for matching)
    vectors = [f.get('desc', '').lower() for f in res.top_features]
    vector_str = " ".join(vectors)

    bias = res.bias
    score = res.score

    # 2. Define Narratives based on Context
    narrative = ""

    # HIGH CONVICTION SCENARIOS
    if score >= 75:
        if bias == "LONG":
            if "volume" in vector_str or "whale" in vector_str:
                narrative = "Aggressive institutional absorption detected at lows. Supply is exhausted."
            elif "breakout" in vector_str or "momentum" in vector_str:
                narrative = "High-velocity breakout in progress. Order flow is heavily one-sided."
            else:
                narrative = "Market structure has shifted bullish across multiple timeframes. dips are for buying."
        elif bias == "SHORT":
            if "volume" in vector_str:
                narrative = "Heavy distribution spotted. Smart money is offloading into retail strength."
            else:
                narrative = "Technical structure has broken down. Expect lower prices as stops get triggered."

    # MODERATE SCENARIOS
    elif score >= 50:
        if bias == "LONG":
            if "rsi" in vector_str:
                narrative = "Momentum divergence suggests sellers are losing control. Reversal likely."
            else:
                narrative = "Price is grinding higher with constructive support building."
        elif bias == "SHORT":
            narrative = "Rally is stalling at key resistance. Momentum is fading."

    # WEAK/CHOPPY SCENARIOS
    else:
        narrative = "Liquidity is thin and direction is unclear. Expect stop-hunts on both sides."

    # Fallback if no specific condition met
    if not narrative:
        narrative = f"Quantitative models indicate a {bias.lower()} lean, but conviction is low. Wait for confirmation."

    return narrative


# ==============================================================================
#  HTMX PARTIALS
# ==============================================================================

@require_http_methods(["GET"])
def hx_ticker(request):
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    data = []
    for s in symbols:
        try:
            df = MarketService.get_historical_data(f"{s}USDT", "PERP", "SCALP")
            if df is not None and not df.empty:
                price = float(df['close'].iloc[-1])
                if len(df) > 1:
                    prev = float(df['close'].iloc[-2])
                    direction = 'up' if price > prev else ('down' if price < prev else 'flat')
                else:
                    direction = 'flat'

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
    if not has_access(request.user):
        return HttpResponse('<div class="text-center text-red-500 font-bold p-10">UPGRADE TO ACCESS</div>')

    symbol = request.POST.get("symbol", "").upper().strip()
    if not symbol: return HttpResponse("")
    if not symbol.endswith("USDT"): symbol += "USDT"

    try:
        mode = request.POST.get("mode", "INTRADAY")
        df = MarketService.get_historical_data(symbol, "PERP", mode)
        if df is None or df.empty: return HttpResponse('<div class="text-red-500 p-4">DATA ERROR</div>')

        # 1. Run Physics Engine
        engine = CryptoQuantEngine()
        res = engine.analyze(df, mode)

        # 2. Generate Tag (Short & Punchy)
        if res.score >= 80:
            note_tag = "INSTITUTIONAL FLOW"
        elif res.score >= 60:
            note_tag = "TREND CONTINUATION"
        elif res.score <= 30:
            note_tag = "CHOPPY / RANGE"
        else:
            note_tag = "POSSIBLE REVERSAL"

        # 3. Generate Narrative (The "Why" - not just the "What")
        note_msg = generate_market_narrative(res)

        return render(request, 'core/partials/dashboard.html', {
            'res': res,
            'symbol': symbol,
            'mode': mode,
            'note_tag': note_tag,
            'note_msg': note_msg
        })
    except Exception as e:
        return HttpResponse(f'<div class="text-red-500 p-4">ERROR: {str(e)}</div>')


@login_required
@require_http_methods(["POST"])
def hx_backtest(request):
    try:
        df = MarketService.get_historical_data(request.POST.get("symbol"), "PERP", "INTRADAY")
        if df is None: return HttpResponse("No Data")
        engine = CryptoBacktestEngine(df, request.POST.get("symbol"))
        stats = engine.run("INTRADAY")
        return render(request, 'core/partials/backtest_result.html', {'stats': stats})
    except Exception as e:
        return HttpResponse(f"Error: {e}")


@login_required
@require_http_methods(["GET"])
def hx_alpha_scan(request):
    if not has_access(request.user): return HttpResponse(
        '<div class="text-center text-red-500 font-bold p-10">ACCESS DENIED</div>')
    vip_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    engine = CryptoQuantEngine()
    results = []

    def scan(sym):
        try:
            df = MarketService.get_historical_data(sym, "PERP", "INTRADAY")
            if df.empty: return None
            res = engine.analyze(df, "INTRADAY")
            if res.bias in ["LONG", "SHORT"] and res.score >= 70:
                return {'symbol': sym, 'bias': res.bias, 'score': res.score, 'entry': res.entry, 'stop': res.stop,
                        'target': res.target1, 'explanation': getattr(res, 'narrative', 'Strong Momentum')}
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
    try:
        JournalEntry.objects.create(
            user=request.user, symbol=request.POST.get('symbol'),
            bias=request.POST.get('bias'), entry_price=float(request.POST.get('entry', 0)),
            stop_loss=float(request.POST.get('stop', 0)), target=float(request.POST.get('target', 0)),
            confidence=float(request.POST.get('score', 0)), status='PENDING'
        )
        return HttpResponse(
            '<button class="w-full mt-8 px-6 py-4 bg-emerald-500/10 text-emerald-400 font-black rounded-xl text-xs uppercase tracking-[0.15em] border border-emerald-500/50 cursor-default flex items-center justify-center gap-2 shadow-lg"><i data-lucide="check" class="w-4 h-4"></i> SAVED TO JOURNAL</button>')
    except:
        return HttpResponse(
            '<button class="w-full mt-8 px-6 py-4 bg-red-500/10 text-red-500 font-bold rounded-xl text-xs uppercase tracking-widest border border-red-500/50">ERROR SAVING</button>')


@login_required
def refresh_journal_entry(request, entry_id):
    entry = get_object_or_404(JournalEntry, id=entry_id, user=request.user)
    msg_type, title, message = "info", "Market Data", "Synced."

    if entry.status == 'PENDING':
        try:
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
            if df is not None and not df.empty:
                current_price = float(df['close'].iloc[-1])
                new_status = 'PENDING'

                # Check Long
                if entry.bias == 'LONG':
                    if current_price >= entry.target:
                        new_status = 'WIN'
                    elif current_price <= entry.stop_loss:
                        new_status = 'LOSS'
                # Check Short
                elif entry.bias == 'SHORT':
                    if current_price <= entry.target:
                        new_status = 'WIN'
                    elif current_price >= entry.stop_loss:
                        new_status = 'LOSS'

                if new_status != 'PENDING':
                    entry.status = new_status
                    entry.exit_price = current_price
                    entry.closed_at = timezone.now()
                    entry.save()
                    msg_type = "success" if new_status == 'WIN' else "warning"
                    title = f"Closed: {new_status}"
                    message = f"Exit: ${current_price}"
                else:
                    message = f"Current: ${current_price}"
        except Exception as e:
            msg_type, title = "error", "Sync Failed"
            log.error(f"Journal Sync Error: {e}")

    response = render(request, 'core/partials/journal_row.html', {'entry': entry})
    response['HX-Trigger'] = json.dumps({'showJournalToast': {'type': msg_type, 'title': title, 'message': message}})
    return response


@login_required
@require_http_methods(["DELETE"])
def delete_journal_entry(request, entry_id):
    JournalEntry.objects.filter(id=entry_id, user=request.user).delete()
    response = HttpResponse("")
    response['HX-Trigger'] = json.dumps(
        {'showJournalToast': {'type': 'success', 'title': 'Deleted', 'message': 'Entry removed.'}})
    return response


def cron_scan_trigger(request, secret_key):
    if secret_key != getattr(settings, 'CRON_SECRET', 'reelioo_master_key'):
        return JsonResponse({'status': 'forbidden'}, status=403)

    logs = []
    # Cleanup
    JournalEntry.objects.filter(created_at__lt=timezone.now() - timedelta(days=90)).delete()

    # Scan
    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']
    active_users = User.objects.filter(
        Q(is_superuser=True) | Q(profile__subscription_status__in=['active', 'on_trial'])).distinct()
    engine = CryptoQuantEngine()

    for symbol in watchlist:
        try:
            df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
            if df is None: continue
            res = engine.analyze(df, "INTRADAY")
            if res.score >= 65 and res.bias in ['LONG', 'SHORT']:
                # Global Debounce (Master Admin)
                admin = User.objects.filter(is_superuser=True).first()
                if admin and not JournalEntry.objects.filter(user=admin, symbol=symbol, status='PENDING',
                                                             created_at__gte=timezone.now() - timedelta(
                                                                     hours=4)).exists():
                    send_discord_alert(symbol, alert_type="SNIPER")
                    for user in active_users:
                        if not JournalEntry.objects.filter(user=user, symbol=symbol, status='PENDING').exists():
                            JournalEntry.objects.create(user=user, symbol=symbol, bias=res.bias, entry_price=res.entry,
                                                        stop_loss=res.stop, target=res.target1, confidence=res.score,
                                                        status='PENDING')
        except:
            continue

    return JsonResponse({'status': 'ok'})


@login_required
def journal_view(request):
    entries = JournalEntry.objects.filter(user=request.user).order_by('-created_at')
    net_roi = 0.0
    g_profit, g_loss = 0.0, 0.0
    for t in entries:
        if t.status in ['WIN', 'LOSS']:
            try:
                # Assuming 1:2 RR for simple calculation or use real prices
                # Here we use real R-multiples based on target/stop distances
                dist_target = abs(t.target - t.entry_price)
                dist_stop = abs(t.entry_price - t.stop_loss)
                r_multiple = dist_target / dist_stop if dist_stop > 0 else 1.0

                if t.status == 'WIN':
                    net_roi += r_multiple
                    g_profit += r_multiple
                elif t.status == 'LOSS':
                    net_roi -= 1.0  # Loss is always -1R
                    g_loss += 1.0
            except:
                continue

    display_roi = round(net_roi, 2)
    pf = round(g_profit / g_loss, 2) if g_loss > 0 else (round(g_profit, 2) if g_profit > 0 else 0)

    paginator = Paginator(entries, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/journal.html', {'page_obj': page_obj, 'net_roi': display_roi, 'profit_factor': pf,
                                                 'active_pending': entries.filter(status='PENDING').count()})


# --- AUTH & SYSTEM ---
def login_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == "POST":
        try:
            user_obj = User.objects.get(email__iexact=request.POST.get('email'))
            user = authenticate(request, username=user_obj.username, password=request.POST.get('password'))
            if user:
                login(request, user); return redirect('terminal')
            else:
                messages.error(request, "Invalid password.")
        except:
            messages.error(request, "User not found.")
    return render(request, 'core/auth/login.html')


def signup_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('pricing')
    else:
        form = SignupForm()
    return render(request, 'core/auth/signup.html', {'form': form})


def logout_view(request): logout(request); return redirect('landing')


def landing_view(request): return redirect('terminal') if request.user.is_authenticated else render(request,
                                                                                                    'core/landing.html')


@login_required
def terminal_view(request): return render(request, 'core/terminal.html')


@login_required
def settings_view(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid(): form.save(); return render(request, 'core/auth/profile_saved.html')
    else:
        form = UserUpdateForm(instance=request.user)
    return render(request, 'core/auth/settings.html', {'form': form})


@login_required
def ops_dashboard_view(request): return render(request,
                                               'core/ops_dashboard.html') if request.user.is_superuser else redirect(
    'terminal')


def terms_view(request): return render(request, 'core/legal/terms.html')


def about_view(request): return render(request, 'core/why_reelioo.html')


def privacy_view(request): return render(request, 'core/legal/privacy.html')


def refund_view(request): return render(request, 'core/legal/refund.html')


def contact_view(request): return render(request, 'core/legal/contact.html')


def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


def robots_view(request): return HttpResponse("User-agent: *\nDisallow:", content_type="text/plain")


def sitemap_view(request): return HttpResponse("", content_type="application/xml")


@login_required
def add_journal_entry(request): return JsonResponse({'status': 'ok'})


# --- CSV SEARCH ---
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
    if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    return JsonResponse(symbols, safe=False)


def search_crypto_view(request):
    return JsonResponse(MarketService.search_assets(request.GET.get("q", "")), safe=False)


# --- LEMON SQUEEZY (WEBHOOK & PORTAL) ---
@csrf_exempt
@require_http_methods(["POST"])
def lemon_squeezy_webhook(request):
    # Signature Check
    signature = request.headers.get("X-Signature")
    if not signature or not LS_SIGNING_SECRET: return HttpResponse("No Sig", status=403)
    digest = hmac.new(LS_SIGNING_SECRET.encode('utf-8'), request.body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, signature): return HttpResponse("Bad Sig", status=403)

    try:
        data = json.loads(request.body)
        meta = data.get('meta', {})
        attrs = data.get('data', {}).get('attributes', {})

        # User Match
        custom = attrs.get('checkout_data', {}).get('custom', {})
        user_id = custom.get('user_id')
        email = attrs.get('user_email')

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

        if user and meta.get('event_name') in ['subscription_created', 'subscription_updated', 'subscription_cancelled',
                                               'subscription_expired', 'subscription_resumed']:
            p = user.profile
            p.lemon_squeezy_subscription_id = data.get('data', {}).get('id')
            p.subscription_status = attrs.get('status')
            p.update_payment_url = attrs.get('urls', {}).get('update_payment_method')
            renews = attrs.get('renews_at') or attrs.get('ends_at')
            if renews: p.subscription_end_date = parser.parse(renews)
            p.save()

        return HttpResponse("OK")
    except:
        return HttpResponse("Err", status=500)


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