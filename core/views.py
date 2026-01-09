import os
import csv
import hmac
import hashlib
import json
import logging
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
from .services.news_service import NewsService
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

    # --- 1. GENERATE DEEP LINK ---
    terminal_link = f"https://reelioo.app/terminal?ticker={symbol}"

    # --- 2. DESIGN THE TEASER ---
    if alert_type == "SNIPER":
        color = 5814783  # Blurple
        title = f"🎯 SNIPER TARGET IDENTIFIED: {symbol}"
        description = "**Institutional Activity Detected.**\nNeural engines have locked onto a high-probability setup."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/3121/3121575.png"
    elif alert_type == "WIN":
        color = 5763719  # Green
        title = f"✅ TARGET HIT: {symbol}"
        description = "Trade closed in profit."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/190/190411.png"
    elif alert_type == "LOSS":
        color = 15548997  # Red
        title = f"🛑 STOP HIT: {symbol}"
        description = "Trade closed at stop loss."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/1828/1828843.png"
    else:
        color = 16776960  # Yellow
        title = f"📡 RADAR CONTACT: {symbol}"
        description = "Volatility Spike Detected."
        thumbnail = "https://cdn-icons-png.flaticon.com/512/564/564619.png"

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
                {"name": "Asset", "value": f"`{symbol}`", "inline": True},
                {"name": "Signal Strength", "value": "██████▒▒▒▒ **[ACTIVE]**", "inline": True},
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

# ==============================================================================
#  LEMON SQUEEZY WEBHOOK (Source of Truth)
# ==============================================================================

@csrf_exempt
@require_http_methods(["POST"])
def lemon_squeezy_webhook(request):
    """Syncs Lemon Squeezy status to local DB"""
    signature = request.headers.get("X-Signature")
    if not signature or not LS_SIGNING_SECRET:
        return HttpResponse("Signature Missing", status=403)

    digest = hmac.new(LS_SIGNING_SECRET.encode('utf-8'), request.body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(digest, signature):
        return HttpResponse("Invalid Signature", status=403)

    try:
        payload = json.loads(request.body)
        data = payload.get('data', {})
        attributes = data.get('attributes', {})

        custom_data = attributes.get('checkout_data', {}).get('custom', {})
        user_id = custom_data.get('user_id')
        email = attributes.get('user_email')

        user = None
        if user_id:
            try: user = User.objects.get(id=user_id)
            except User.DoesNotExist: pass
        if not user and email:
            try: user = User.objects.get(email__iexact=email)
            except User.DoesNotExist: pass

        if not user:
            return HttpResponse("User not found", status=200)

        profile = user.profile
        event_name = payload.get('meta', {}).get('event_name')

        if event_name in ['subscription_created', 'subscription_updated', 'subscription_cancelled', 'subscription_expired', 'subscription_resumed']:
            profile.lemon_squeezy_subscription_id = data.get('id')
            profile.lemon_squeezy_customer_id = attributes.get('customer_id')
            profile.subscription_status = attributes.get('status')
            profile.update_payment_url = attributes.get('urls', {}).get('update_payment_method')

            renews_at_str = attributes.get('renews_at') or attributes.get('ends_at')
            if renews_at_str:
                profile.renews_at = parser.parse(renews_at_str)

            profile.save()
            log.info(f"Updated Subscription for {user.username}: {profile.subscription_status}")

        return HttpResponse("Webhook Processed")
    except Exception as e:
        log.error(f"Webhook Error: {e}")
        return HttpResponse("Server Error", status=500)

# ==============================================================================
#  BILLING VIEWS
# ==============================================================================

@login_required
def pricing_view(request):
    checkout_url = f"https://reelioo.lemonsqueezy.com/checkout/buy/{LS_VARIANT_ID}?"
    checkout_url += f"checkout[email]={request.user.email}&"
    checkout_url += f"checkout[custom][user_id]={request.user.id}"

    context = {
        "checkout_url": checkout_url,
        "is_premium": request.user.profile.is_premium
    }
    return render(request, 'core/pricing.html', context)

@login_required
def billing_portal_view(request):
    sub_id = request.user.profile.lemon_squeezy_subscription_id
    if not sub_id:
        messages.error(request, "No active subscription found.")
        return redirect('pricing')

    try:
        url = f"https://api.lemonsqueezy.com/v1/subscriptions/{sub_id}"
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
            "Authorization": f"Bearer {LS_API_KEY}"
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        customer_portal_url = data['data']['attributes']['urls']['customer_portal']
        return redirect(customer_portal_url)
    except Exception as e:
        log.error(f"Portal Error: {e}")
        messages.error(request, "Could not load billing portal.")
        return redirect('settings')

# ==============================================================================
#  HTMX PARTIALS & CORE APP
# ==============================================================================

@require_http_methods(["GET"])
def hx_ticker(request):
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    data = []
    for s in symbols:
        try:
            # Fetch data
            df = MarketService.get_historical_data(f"{s}USDT", "PERP", "SCALP")

            if df is not None and not df.empty:
                price = float(df['close'].iloc[-1])

                # Logic: Compare current price with previous close to determine color
                if len(df) > 1:
                    prev_price = float(df['close'].iloc[-2])
                else:
                    prev_price = price  # No history, assume neutral

                if price > prev_price:
                    direction = 'up'  # Green
                elif price < prev_price:
                    direction = 'down'  # Red
                else:
                    direction = 'flat'  # White

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
        engine = CryptoQuantEngine()
        res = engine.analyze(df, mode)
        raw_note = NewsService.get_smart_insights(symbol, mode)
        note_tag, note_msg = raw_note.split("|", 1) if "|" in raw_note else ("INSIGHT", raw_note)

        return render(request, 'core/partials/dashboard.html', {
            'res': res, 'symbol': symbol, 'mode': mode,
            'note_tag': note_tag.strip(), 'note_msg': note_msg.strip()
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
    if not has_access(request.user): return HttpResponse('<div class="text-center text-red-500 font-bold p-10">ACCESS DENIED</div>')
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
        except: return None

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
        # FIX: Added 'mt-8' to prevent jumping. Matches the original form's margin.
        return HttpResponse(
            '<button class="w-full mt-8 px-6 py-4 bg-emerald-500/10 text-emerald-400 font-black rounded-xl text-xs uppercase tracking-[0.15em] border border-emerald-500/50 cursor-default flex items-center justify-center gap-2 shadow-lg"><i data-lucide="check" class="w-4 h-4"></i> SAVED TO JOURNAL</button>')
    except:
        return HttpResponse(
            '<button class="w-full mt-8 px-6 py-4 bg-red-500/10 text-red-500 font-bold rounded-xl text-xs uppercase tracking-widest border border-red-500/50">ERROR SAVING</button>')

    # --- JOURNAL ACTIONS ---

# --- JOURNAL ACTIONS ---

@login_required
def refresh_journal_entry(request, entry_id):
    """
    Checks the specific trade against current market data.
    Updates status (WIN/LOSS/PENDING) and renders the row row for HTMX.
    """
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
                    if current_price >= entry.target: new_status = 'WIN'
                    elif current_price <= entry.stop_loss: new_status = 'LOSS'
                # Check Short
                elif entry.bias == 'SHORT':
                    if current_price <= entry.target: new_status = 'WIN'
                    elif current_price >= entry.stop_loss: new_status = 'LOSS'

                if new_status != 'PENDING':
                    entry.status = new_status
                    entry.exit_price = current_price
                    entry.closed_at = timezone.now()
                    entry.save()
                    msg_type = "success" if new_status == 'WIN' else "warning"
                    title = f"Closed: {new_status}"
                    message = f"Exit: ${current_price}"
                    # Optional: send_discord_alert(f"{entry.symbol} {new_status}", alert_type=new_status)
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
    response['HX-Trigger'] = json.dumps({'showJournalToast': {'type': 'success', 'title': 'Deleted', 'message': 'Entry removed.'}})
    return response

# --- CRON JOB (Signal Engine) ---

def cron_scan_trigger(request, secret_key):
    # 1. SECURITY CHECK
    required_secret = getattr(settings, 'CRON_SECRET', 'reelioo_master_key')
    if secret_key != required_secret:
        return JsonResponse({'status': 'forbidden', 'message': 'Access Denied'}, status=403)

    logs = []

    # 2. CLEANUP OLD DATA
    retention_cutoff = timezone.now() - timedelta(days=90)
    deleted_count, _ = JournalEntry.objects.filter(created_at__lt=retention_cutoff).delete()
    if deleted_count > 0: logs.append(f"Purged {deleted_count} entries.")

    # 3. SCAN LOGIC
    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']
    scanned = 0
    signals_sent = 0
    engine = CryptoQuantEngine()

    # Find users who should receive signals (Admins + Active/Trial Subs)
    active_users = User.objects.filter(
        Q(is_superuser=True) |
        Q(profile__subscription_status__in=['active', 'on_trial'])
    ).distinct()

    for symbol in watchlist:
        try:
            df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
            if df is None or df.empty: continue

            res = engine.analyze(df, "INTRADAY")

            # TRIGGER: High Score + Directional
            if res.score >= 65 and res.bias in ['LONG', 'SHORT']:
                # Global Debounce (Check master admin)
                master_admin = User.objects.filter(is_superuser=True).first()
                recent_exists = False
                if master_admin:
                    recent_exists = JournalEntry.objects.filter(
                        user=master_admin, symbol=symbol, status='PENDING',
                        created_at__gte=timezone.now() - timedelta(hours=4)
                    ).exists()

                if not recent_exists:
                    # Alert Discord
                    send_discord_alert(symbol, alert_type="SNIPER")

                    # Distribute to Users
                    for user in active_users:
                        if not JournalEntry.objects.filter(user=user, symbol=symbol, status='PENDING').exists():
                            JournalEntry.objects.create(
                                user=user, symbol=symbol, bias=res.bias,
                                entry_price=res.entry, stop_loss=res.stop, target=res.target1,
                                confidence=res.score, status='PENDING', leverage='Low'
                            )
                    signals_sent += 1
                    logs.append(f"Signal: {symbol} -> {active_users.count()} users")

        except Exception as e:
            logs.append(f"Error {symbol}: {str(e)}")
            continue
        scanned += 1

    return JsonResponse({
        'status': 'success',
        'scanned': scanned,
        'signals_distributed': signals_sent,
        'logs': logs
    })

# --- PAGE VIEWS & CSV SEARCH ---

@login_required
def journal_view(request):
    entries = JournalEntry.objects.filter(user=request.user).order_by('-created_at')

    # --- ROI CALCULATION ---
    net_roi = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in entries:
        if trade.status in ['WIN', 'LOSS']:
            try:
                # Calculate % movement
                if trade.bias == 'LONG':
                    potential_win = (trade.target - trade.entry_price) / trade.entry_price
                    potential_loss = (trade.entry_price - trade.stop_loss) / trade.entry_price
                else:  # SHORT
                    potential_win = (trade.entry_price - trade.target) / trade.entry_price
                    potential_loss = (trade.stop_loss - trade.entry_price) / trade.entry_price

                if trade.status == 'WIN':
                    net_roi += potential_win
                    gross_profit += potential_win
                elif trade.status == 'LOSS':
                    net_roi -= potential_loss
                    gross_loss += potential_loss
            except: continue

    # Calculate Display Values
    display_roi = round(net_roi * 100, 2)
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0)
    active = entries.filter(status='PENDING').count()

    paginator = Paginator(entries, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    return render(request, 'core/journal.html', {
        'page_obj': page_obj,
        'net_roi': display_roi, # Passed as float for correct template coloring
        'profit_factor': pf,
        'active_pending': active
    })

def login_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == "POST":
        try:
            user_obj = User.objects.get(email__iexact=request.POST.get('email'))
            user = authenticate(request, username=user_obj.username, password=request.POST.get('password'))
            if user: login(request, user); return redirect('terminal')
            else: messages.error(request, "Invalid password.")
        except: messages.error(request, "User not found.")
    return render(request, 'core/auth/login.html')

def signup_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('pricing') # Start Trial
    else: form = SignupForm()
    return render(request, 'core/auth/signup.html', {'form': form})

def logout_view(request): logout(request); return redirect('landing')
def landing_view(request): return redirect('terminal') if request.user.is_authenticated else render(request, 'core/landing.html')
@login_required
def terminal_view(request): return render(request, 'core/terminal.html')
@login_required
def settings_view(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid(): form.save(); return render(request, 'core/auth/profile_saved.html')
    else: form = UserUpdateForm(instance=request.user)
    return render(request, 'core/auth/settings.html', {'form': form})
@login_required
def ops_dashboard_view(request): return render(request, 'core/ops_dashboard.html') if request.user.is_superuser else redirect('terminal')
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

# --- CSV SEARCH (RESTORED) ---
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
        except: pass
    if not symbols: symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    return JsonResponse(symbols, safe=False)

def search_crypto_view(request):
    q = request.GET.get("q", "")
    return JsonResponse(MarketService.search_assets(q), safe=False)