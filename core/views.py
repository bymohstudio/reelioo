import os
import json
import logging
import requests
import concurrent.futures
import razorpay  # REQUIRED for payments
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
from .models import JournalEntry
from .forms import UserUpdateForm, SignupForm

log = logging.getLogger(__name__)


# --- HELPERS ---
def send_discord_alert(message, type="info"):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url: return
    color = 3447003
    if type == "win":
        color = 5763719
    elif type == "loss":
        color = 15548997
    elif type == "new":
        color = 15105570
    payload = {"embeds": [{"title": "Reelioo Intelligence", "description": message, "color": color,
                           "footer": {"text": "Institutional Market Structure"}}]}
    try:
        requests.post(webhook_url, json=payload, timeout=3)
    except:
        pass


def has_access(user):
    if user.profile.is_premium: return True
    return timezone.now() < (user.date_joined + timedelta(days=21))


# ==============================================================================
#  HTMX PARTIALS
# ==============================================================================

@require_http_methods(["GET"])
def hx_ticker(request):
    """Fetches LIVE prices for the marquee ticker."""
    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    data = []
    for s in symbols:
        try:
            df = MarketService.get_historical_data(f"{s}USDT", "PERP", "SCALP")
            if df is not None and not df.empty:
                price = float(df['close'].iloc[-1])
                fmt_price = f"${price:,.2f}" if price > 1.0 else f"${price:,.4f}"
                data.append({'symbol': s, 'price': fmt_price})
            else:
                data.append({'symbol': s, 'price': "---"})
        except Exception:
            data.append({'symbol': s, 'price': "---"})
    return render(request, 'core/partials/ticker.html', {'ticker': data * 4})


@login_required
@require_http_methods(["POST"])
def hx_analyze(request):
    if not has_access(request.user):
        return HttpResponse('<div class="text-center text-red-500 font-bold p-10">TRIAL EXPIRED</div>')

    symbol = request.POST.get("symbol", "").upper().strip()
    if not symbol: return HttpResponse("")

    if not symbol.endswith("USDT"): symbol += "USDT"
    mode = request.POST.get("mode", "INTRADAY")

    try:
        df = MarketService.get_historical_data(symbol, "PERP", mode)
        if df is None or df.empty: return HttpResponse('<div class="text-red-500 p-4">DATA ERROR</div>')
        engine = CryptoQuantEngine()
        res = engine.analyze(df, mode)

        raw_note = NewsService.get_smart_insights(symbol)
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
    symbol = request.POST.get("symbol")
    mode = request.POST.get("mode", "INTRADAY")
    try:
        df = MarketService.get_historical_data(symbol, "PERP", mode)
        if df is None: return HttpResponse("No Data")
        engine = CryptoBacktestEngine(df, symbol)
        stats = engine.run(mode)
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
            '<button class="px-6 py-4 bg-emerald-500/20 text-emerald-400 font-bold rounded-xl text-xs uppercase tracking-widest border border-emerald-500/50 cursor-default">SAVED</button>')
    except:
        return HttpResponse('<button class="text-red-500 font-bold">Error</button>')


# ==============================================================================
#  JOURNAL ACTIONS
# ==============================================================================

@login_required
def refresh_journal_entry(request, entry_id):
    entry = get_object_or_404(JournalEntry, id=entry_id, user=request.user)
    msg_type, title, message = "info", "Market Data", "Synced."
    if entry.status == 'PENDING':
        try:
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
            if df is not None and not df.empty:
                price = float(df['close'].iloc[-1])
                new_s = 'PENDING'
                if entry.bias == 'LONG':
                    if price >= entry.target:
                        new_s = 'WIN'; entry.exit_price = price
                    elif price <= entry.stop_loss:
                        new_s = 'LOSS'; entry.exit_price = price
                elif entry.bias == 'SHORT':
                    if price <= entry.target:
                        new_s = 'WIN'; entry.exit_price = price
                    elif price >= entry.stop_loss:
                        new_s = 'LOSS'; entry.exit_price = price

                if new_s != 'PENDING':
                    entry.status = new_s;
                    entry.exit_price = price;
                    entry.closed_at = timezone.now();
                    entry.save()
                    msg_type = "success" if new_s == 'WIN' else "warning"
                    title = f"Closed: {new_s}";
                    message = f"Exit: ${price}"
                    send_discord_alert(f"Sync: {entry.symbol} Closed as {new_s}", type=new_s.lower())
                else:
                    message = f"Current: ${price}"
        except:
            msg_type, title = "error", "Sync Failed"
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


# ==============================================================================
#  CRON & API
# ==============================================================================

def cron_scan_trigger(request, secret_key):
    if secret_key != os.getenv("CRON_SECRET", "reelioo_master_key"): return JsonResponse({'status': 'forbidden'},
                                                                                         status=403)
    pending = JournalEntry.objects.filter(status='PENDING')
    updated = 0
    for t in pending:
        try:
            df = MarketService.get_historical_data(t.symbol, "PERP", "SCALP")
            if df is None: continue
            price = float(df['close'].iloc[-1])
            new_s = 'PENDING'
            if t.bias == 'LONG':
                if price >= t.target:
                    new_s = 'WIN'
                elif price <= t.stop_loss:
                    new_s = 'LOSS'
            elif t.bias == 'SHORT':
                if price <= t.target:
                    new_s = 'WIN'
                elif price >= t.stop_loss:
                    new_s = 'LOSS'
            if new_s != 'PENDING':
                t.status = new_s;
                t.exit_price = price;
                t.closed_at = timezone.now();
                t.save();
                updated += 1
                send_discord_alert(f"Trade Result: {t.symbol} {new_s}", type=new_s.lower())
        except:
            continue
    return JsonResponse({'status': 'ok', 'updated': updated})


@login_required
def global_symbols_view(request):
    """API for search bar"""
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
    q = request.GET.get("q", "")
    return JsonResponse(MarketService.search_assets(q), safe=False)


# ==============================================================================
#  PAGE VIEWS & AUTH
# ==============================================================================

def login_view(request):
    if request.user.is_authenticated: return redirect('terminal')

    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            # Login via Email
            user_obj = User.objects.get(email__iexact=email)
            user = authenticate(request, username=user_obj.username, password=password)
            if user:
                login(request, user)
                return redirect('terminal')
            else:
                messages.error(request, "Invalid password.")
        except User.DoesNotExist:
            messages.error(request, "No account found with this email.")

    return render(request, 'core/auth/login.html')


def signup_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('terminal')
    else:
        form = SignupForm()
    return render(request, 'core/auth/signup.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('landing')


def landing_view(request):
    if request.user.is_authenticated: return redirect('terminal')
    return render(request, 'core/landing.html')


@login_required
def terminal_view(request):
    return render(request, 'core/terminal.html', {'days_left': 21, 'is_premium': request.user.profile.is_premium})


@login_required
def settings_view(request):
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid(): form.save(); return render(request, 'core/auth/profile_saved.html')
    else:
        form = UserUpdateForm(instance=request.user)
    return render(request, 'core/auth/settings.html', {'form': form})


@login_required
def journal_view(request):
    entries = JournalEntry.objects.filter(user=request.user).order_by('-created_at')
    wins = entries.filter(status='WIN').count()
    losses = entries.filter(status='LOSS').count()
    active = entries.filter(status='PENDING').count()
    net_roi = (wins * 2) - (losses * 1)
    pf = round((wins * 2) / (losses * 1), 2) if losses > 0 else (wins * 2)
    paginator = Paginator(entries, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'core/journal.html',
                  {'page_obj': page_obj, 'net_roi': net_roi, 'profit_factor': pf, 'active_pending': active})


@login_required
def ops_dashboard_view(request):
    if not request.user.is_superuser: return redirect('terminal')
    return render(request, 'core/ops_dashboard.html', {'total_users': User.objects.count(),
                                                       'active_subs': User.objects.filter(
                                                           profile__is_premium=True).count(),
                                                       'total_signals': JournalEntry.objects.count(),
                                                       'recent_signals': JournalEntry.objects.order_by('-created_at')[
                                                                         :10], 'win_rate': 68.5})


# --- PRICING & PAYMENT LOGIC ---

@login_required
def pricing_view(request):
    key_id = os.getenv("RAZORPAY_KEY_ID")
    plan_id = os.getenv("RAZORPAY_PLAN_ID")

    # Debug: Check logs if payment popup fails
    if not key_id or not plan_id:
        log.error("PAYMENT ERROR: Missing RAZORPAY_KEY_ID or RAZORPAY_PLAN_ID in .env")

    context = {
        "key_id": key_id,
        "sub_id": plan_id or "sub_default_placeholder",
        "user_email": request.user.email
    }
    return render(request, 'core/pricing.html', context)


@csrf_exempt
def payment_success_view(request):
    """Handles the success callback from Razorpay"""
    if request.method == "POST":
        try:
            client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

            data = {
                'razorpay_payment_id': request.POST.get('razorpay_payment_id'),
                'razorpay_subscription_id': request.POST.get('razorpay_subscription_id'),
                'razorpay_signature': request.POST.get('razorpay_signature')
            }

            # Verify Signature
            client.utility.verify_subscription_payment_signature(data)

            # Upgrade User
            if request.user.is_authenticated:
                profile = request.user.profile
                profile.is_premium = True
                profile.subscription_status = 'active'
                profile.subscription_id = data['razorpay_subscription_id']
                profile.save()
                messages.success(request, "Reelioo Pro Activated Successfully!")
                return render(request, 'core/auth/success.html')  # Render the Success Splash Page

        except Exception as e:
            log.error(f"Payment Verification Failed: {str(e)}")
            messages.error(request, "Payment verification failed. Please contact support.")
            return redirect('pricing')

    return redirect('pricing')


@login_required
def cancel_subscription_view(request): return redirect('settings')


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