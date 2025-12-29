import razorpay
import os
import json
import requests
import logging
import traceback
from datetime import datetime, timedelta

from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone
from django.utils.html import strip_tags
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.conf import settings

from .quant.crypto_engine import CryptoQuantEngine
from .services.marketdata_service import MarketService
from core.utils import analyze_market_data

import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# --- PUBLIC PAGES ---
def landing_view(request):
    if request.user.is_authenticated:
        return redirect('terminal')
    return render(request, 'core/landing.html')


# --- AUTHENTICATION ---
def signup_view(request):
    from .forms import SignupForm

    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            try:
                subject = "Access Granted: Reelioo Neural Terminal Online"
                html_message = f"""
                <!DOCTYPE html>
                <html><body style="background-color: #000; font-family: sans-serif; color: #ccc;">
                    <div style="max-width: 600px; margin: auto; background: #050505; border: 1px solid #333; padding: 40px;">
                        <h2 style="color: #fff;">REEL<span style="color: #2563eb;">IOO</span></h2>
                        <h1 style="color: #fff;">Protocol Initialized.</h1>
                        <p>Hello <strong>{user.username}</strong>,</p>
                        <p>Your access to the Reelioo Neural Engine is confirmed. For the next 21 days, you have the visibility of an institutional trading desk.</p>
                        <br>
                        <a href="https://reelioo.app/login" style="background: #2563eb; color: #fff; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">LAUNCH TERMINAL</a>
                        <br><br>
                        <p><strong>Mission Directive:</strong><br>1. Go to Terminal.<br>2. Type 'BTC'.<br>3. Decode the Order Flow.</p>
                        <hr style="border-color: #333;">
                        <p style="text-align: center; color: #fff; font-weight: bold;">Trade Less. Win More.</p>
                    </div>
                </body></html>"""
                plain_message = strip_tags(html_message)
                send_mail(subject, plain_message, settings.DEFAULT_FROM_EMAIL, [user.email], html_message=html_message,
                          fail_silently=True)
                send_mail(f"🚀 New Signup: {user.username}", f"Email: {user.email}", settings.DEFAULT_FROM_EMAIL,
                          ['reeliooapp@gmail.com'], fail_silently=True)
            except Exception as e:
                print(f"⚠️ Email System Error: {e}")

            login(request, user)
            messages.success(request, "Account Initialized. 21-Day Trial Active.")
            return redirect('terminal')
    else:
        form = SignupForm()
    return render(request, 'core/auth/signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            from django.contrib.auth.models import User
            user_obj = User.objects.get(email=email)
            user = authenticate(username=user_obj.username, password=password)
            if user is not None:
                login(request, user)
                return redirect('terminal')
            else:
                messages.error(request, "Invalid credentials.")
        except Exception:
            messages.error(request, "No account found.")
    return render(request, 'core/auth/login.html')


def logout_view(request):
    logout(request)
    return redirect('landing')


# --- TERMINAL ---
@login_required(login_url='login')
def terminal_view(request):
    profile = request.user.profile
    if not profile.is_access_granted():
        messages.warning(request, "Trial Expired. Please Upgrade to Access Terminal.")
        return redirect('pricing')
    context = {'days_left': profile.get_days_left(), 'is_premium': profile.is_premium,
               'username': request.user.username}
    return render(request, 'core/terminal.html', context)


# --- PRICING ---
@login_required
def pricing_view(request):
    profile = request.user.profile
    if profile.subscription_status == 'cancellation_pending':
        if profile.is_access_granted():
            messages.info(request, "You have an active plan. Wait for it to expire before resubscribing.")
            return redirect('settings')

    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    plan_id = os.getenv("RAZORPAY_PLAN_ID")
    client = razorpay.Client(auth=(key_id, key_secret))
    sub_id = "error"

    try:
        if key_id and key_secret and plan_id:
            subscription = client.subscription.create(
                {"plan_id": plan_id, "total_count": 60, "quantity": 1, "customer_notify": 1,
                 "notes": {"email": request.user.email}})
            sub_id = subscription['id']
            profile.razorpay_subscription_id = sub_id
            profile.save()
    except Exception as e:
        print(f"Razorpay Init Error: {e}")

    context = {"key_id": key_id, "sub_id": sub_id, "user_email": request.user.email,
               "is_trial_expired": profile.is_trial_expired() and not profile.is_premium}
    return render(request, 'core/pricing.html', context)


# --- PAYMENT SUCCESS ---
@csrf_exempt
def payment_success_view(request):
    from .models import UserProfile
    if request.method == "POST":
        try:
            payment_id = request.POST.get('razorpay_payment_id')
            subscription_id = request.POST.get('razorpay_subscription_id')
            signature = request.POST.get('razorpay_signature')
            client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
            data_to_verify = {'razorpay_payment_id': payment_id, 'razorpay_subscription_id': subscription_id,
                              'razorpay_signature': signature}
            client.utility.verify_subscription_payment_signature(data_to_verify)
            try:
                profile = UserProfile.objects.get(razorpay_subscription_id=subscription_id)
                user = profile.user
                profile.is_premium = True
                profile.subscription_status = "active"
                profile.subscription_end_date = timezone.now() + timedelta(days=30)
                profile.save()
                user.backend = 'django.contrib.auth.backends.ModelBackend'
                login(request, user)
                return render(request, 'core/success.html')
            except UserProfile.DoesNotExist:
                return redirect('pricing')
        except Exception as e:
            return render(request, 'core/payment_failed.html')
    return redirect('pricing')


# --- CANCEL SUBSCRIPTION ---
@login_required
def cancel_subscription_view(request):
    if request.method == "POST":
        profile = request.user.profile
        sub_id = profile.razorpay_subscription_id
        if sub_id and profile.is_premium:
            try:
                client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
                sub_details = client.subscription.fetch(sub_id)
                current_end_timestamp = sub_details.get('current_end')
                if current_end_timestamp:
                    end_date = datetime.fromtimestamp(current_end_timestamp)
                    end_date = timezone.make_aware(end_date)
                    profile.subscription_end_date = end_date
                else:
                    if not profile.subscription_end_date:
                        profile.subscription_end_date = timezone.now() + timedelta(days=30)
                client.subscription.cancel(sub_id, {'cancel_at_cycle_end': 1})
                profile.subscription_status = "cancellation_pending"
                profile.save()
                return render(request, 'core/cancel_success.html')
            except Exception as e:
                print(f"Cancel Error: {e}")
                messages.error(request, "Could not cancel. Contact support.")
    return redirect('settings')


# --- SETTINGS ---
@login_required
def settings_view(request):
    from .forms import UserUpdateForm
    user = request.user
    profile = user.profile
    profile.is_access_granted()
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=user)
        if form.is_valid():
            user = form.save()
            new_country = form.cleaned_data.get('country')
            if new_country:
                profile.country = new_country
                profile.save()
                return render(request, 'core/auth/profile_saved.html')
        else:
            messages.error(request, "Update failed.")
    else:
        initial_data = {'country': profile.country}
        form = UserUpdateForm(instance=user, initial=initial_data)
    return render(request, 'core/auth/settings.html', {'form': form})


# --- JOURNAL ---
@login_required
def journal_view(request):
    # --- SECURITY CHECK ---
    profile = request.user.profile
    if not profile.is_access_granted():
        messages.warning(request, "Access Expired. Upgrade to View Journal.")
        return redirect('pricing')
    # ----------------------

    from .models import JournalEntry
    entries_list = JournalEntry.objects.filter(user=request.user).order_by('-created_at')

    # --- NEW: CALCULATE ROI METRICS ---
    net_roi = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    # We iterate through closed trades to calculate performance
    # (Doing this in Python is fine for < 1000 trades. For scaling, use DB aggregation later)
    for trade in entries_list:
        if trade.status in ['WIN', 'LOSS']:
            # Calculate expected % move based on Entry/Target/Stop
            try:
                if trade.bias == 'LONG':
                    potential_win = (trade.target - trade.entry_price) / trade.entry_price
                    potential_loss = (trade.entry_price - trade.stop_loss) / trade.entry_price
                else:  # SHORT
                    potential_win = (trade.entry_price - trade.target) / trade.entry_price
                    potential_loss = (trade.stop_loss - trade.entry_price) / trade.entry_price

                # Add Leverage Multiplier (Simulated 10x for "Low" lev, or just raw %)
                # For now, let's show RAW asset PnL (Safer/Cleaner)

                if trade.status == 'WIN':
                    net_roi += potential_win
                    gross_profit += potential_win
                elif trade.status == 'LOSS':
                    net_roi -= potential_loss
                    gross_loss += potential_loss
            except ZeroDivisionError:
                continue

    # Format for display
    display_roi = round(net_roi * 100, 2)
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
        round(gross_profit, 2) if gross_profit > 0 else 0)

    # Pagination
    paginator = Paginator(entries_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'net_roi': display_roi,  # Replaces Win Rate
        'profit_factor': profit_factor,  # New Metric
        'active_pending': entries_list.filter(status='PENDING').count()
    }
    return render(request, 'core/journal.html', context)


@login_required
def add_journal_entry(request):
    from .models import JournalEntry
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # --- FILTER REMOVED: USER FREEDOM RESTORED ---
            # We accept whatever confidence the user/terminal sends.
            # You are the trader; you decide what goes in the journal.

            JournalEntry.objects.create(
                user=request.user,
                symbol=data.get('symbol'),
                bias=data.get('bias'),
                entry_price=float(data.get('entry')),
                stop_loss=float(data.get('stop')),
                target=float(data.get('target')),
                confidence=float(data.get('confidence', 0)),  # Saves exact score (e.g. 38.5)
                leverage=data.get('leverage', 'Low')
            )
            return JsonResponse({'status': 'success', 'message': 'Trade saved to Journal.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
def refresh_journal_entry(request, entry_id):
    from .models import JournalEntry

    if request.method == "POST":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)

            # 1. GET MARKET DATA (SCALP precision for accuracy)
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
            if df is None or df.empty:
                return JsonResponse({'status': 'error', 'message': 'Market data unavailable'})

            current_price = float(df['close'].iloc[-1])
            new_status = entry.status

            # 2. SIGNAL LIFECYCLE
            now = timezone.now()
            duration = now - entry.created_at
            hours_open = duration.total_seconds() / 3600
            MAX_DURATION = 24  # Keep trades active for 24h before timing out

            # 3. SMART EXPIRATION
            if entry.status in ["PENDING", "SHIELD"] and hours_open > MAX_DURATION:
                if current_price < entry.entry_price:
                    if current_price <= entry.stop_loss:
                        new_status = "LOSS"
                    else:
                        new_status = "TIMEOUT"  # Price didn't move
                else:
                    new_status = "STAGNANT"  # Price moved up but didn't hit target

                entry.status = new_status
                entry.save()
                return JsonResponse({
                    'status': 'success',
                    'new_status': new_status,
                    'message': "Signal Expired.",
                    'confidence': entry.confidence
                })

            # 4. CHECK STANDARD OUTCOMES
            if entry.status == "PENDING":
                if entry.bias == "LONG":
                    if current_price >= entry.target:
                        new_status = "WIN"
                    elif current_price <= entry.stop_loss:
                        new_status = "LOSS"
                elif entry.bias == "SHORT":
                    if current_price <= entry.target:
                        new_status = "WIN"
                    elif current_price >= entry.stop_loss:
                        new_status = "LOSS"

            # 5. CHECK SHIELD OUTCOMES
            elif entry.status == "SHIELD":
                if current_price >= entry.target:
                    new_status = "MISSED"
                elif current_price <= entry.stop_loss:
                    new_status = "SAVED"

            # 6. SAVE UPDATE
            if new_status != entry.status:
                entry.status = new_status
                entry.save()

            return JsonResponse({
                'status': 'success',
                'new_status': new_status,
                'current_price': current_price,
                'confidence': entry.confidence
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=405)


@login_required
def delete_journal_entry(request, entry_id):
    from .models import JournalEntry
    if request.method == "DELETE":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)
            entry.delete()
            return JsonResponse({'status': 'success'})
        except JournalEntry.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not found'}, status=404)


def robots_view(request):
    content = render_to_string('core/robots.txt')
    return HttpResponse(content, content_type="text/plain")


def sitemap_view(request):
    content = render_to_string('core/sitemap.xml')
    return HttpResponse(content, content_type="application/xml")


# --- CRON JOB TRIGGER ---
def cron_scan_trigger(request, secret_key):
    from .models import JournalEntry, UserProfile
    from django.contrib.auth.models import User
    from django.db.models import Q
    from django.utils import timezone
    from datetime import timedelta

    # 1. SECURITY CHECK
    required_secret = getattr(settings, 'CRON_SECRET', 'super-secret-password-123')
    if secret_key != required_secret:
        return JsonResponse({'status': 'forbidden', 'message': 'Access Denied'}, status=403)

    logs = []

    # 2. DATA RETENTION POLICY (CLEANUP)
    # Automatically delete signals older than 90 days to keep DB fast
    retention_cutoff = timezone.now() - timedelta(days=90)
    deleted_count, _ = JournalEntry.objects.filter(created_at__lt=retention_cutoff).delete()
    if deleted_count > 0:
        logs.append(f"🧹 MAINTENANCE: Purged {deleted_count} old entries.")

    # 3. SETUP SCANNER
    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'WIFUSDT']
    scanned = 0
    signals_sent = 0
    engine = CryptoQuantEngine()

    # 4. GET ALL VALID USERS (Admins + Active Subscribers)
    # We find everyone who should receive the trade
    now = timezone.now()
    active_users = User.objects.filter(
        Q(is_superuser=True) |
        Q(profile__subscription_end_date__gt=now)
    ).distinct()

    # 5. START SCAN LOOP
    for symbol in watchlist:
        try:
            # Use PERP data to match Terminal accuracy
            df = MarketService.get_historical_data(symbol, "PERP", "INTRADAY")
            if df is None or df.empty:
                logs.append(f"❌ {symbol}: No Data")
                continue

            res = engine.analyze(df, "INTRADAY")

            # 6. TRIGGER LOGIC (Score > 65% + Directional Bias)
            if res.score >= 65 and res.bias in ['LONG', 'SHORT']:

                # Check Global Duplication (via Admin's Journal as Master Record)
                master_admin = User.objects.filter(is_superuser=True).first()
                recent_signal_exists = False

                if master_admin:
                    recent_signal_exists = JournalEntry.objects.filter(
                        user=master_admin,
                        symbol=symbol,
                        status='PENDING',
                        created_at__gte=timezone.now() - timedelta(hours=4)
                    ).exists()

                if not recent_signal_exists:
                    # A. Send "Teaser" Alert to Discord
                    send_discord_alert(symbol, res, alert_type="SNIPER")

                    # B. DISTRIBUTE TRADE TO ALL ACTIVE USERS
                    # This ensures when they click the link, the trade is ALREADY in their Journal
                    for user in active_users:
                        # Double check specific user doesn't have it
                        user_has_trade = JournalEntry.objects.filter(
                            user=user,
                            symbol=symbol,
                            status__in=['PENDING', 'SHIELD'],
                            created_at__gte=timezone.now() - timedelta(hours=4)
                        ).exists()

                        if not user_has_trade:
                            JournalEntry.objects.create(
                                user=user,
                                symbol=symbol,
                                bias=res.bias,
                                entry_price=res.entry,
                                stop_loss=res.stop,
                                target=res.target1,
                                confidence=res.score,
                                status='PENDING',
                                leverage='Low'
                            )

                    logs.append(f"✅ {symbol}: Signal Distributed to {active_users.count()} users.")
                    signals_sent += 1
                else:
                    logs.append(f"⚠️ {symbol}: Signal Active. Distribution skipped.")

        except Exception as e:
            logs.append(f"💀 {symbol}: Error - {str(e)}")
            continue

        scanned += 1

    return JsonResponse({
        'status': 'success',
        'scanned': scanned,
        'signals_distributed': signals_sent,
        'users_reached': active_users.count(),
        'logs': logs
    })


def send_discord_alert(symbol, data, alert_type="SNIPER"):
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

# --- LEGAL PAGES ---
def terms_view(request): return render(request, 'core/legal/terms.html')


def privacy_view(request): return render(request, 'core/legal/privacy.html')


def refund_view(request): return render(request, 'core/legal/refund.html')


def contact_view(request): return render(request, 'core/legal/contact.html')


def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


def debug_models_view(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quant_dir = os.path.join(base_dir, "quant")
    model_dir = os.path.join(quant_dir, "ml_models")
    report = {"scanned_path": model_dir, "exists": os.path.exists(model_dir), "files_found": [], "load_tests": {}}

    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        for f in files:
            f_path = os.path.join(model_dir, f)
            size_kb = os.path.getsize(f_path) / 1024
            report["files_found"].append(f"{f} ({size_kb:.2f} KB)")
            if size_kb < 2.0:
                report["load_tests"][f] = "⚠️ WARNING: File too small. Likely Git LFS pointer."

    models_to_check = {
        "xgb_long.json": "xgb", "xgb_short.json": "xgb",
        "lgb_long.txt": "lgb", "lgb_short.txt": "lgb",
        "cat_long.cbm": "cat", "cat_short.cbm": "cat"
    }

    for filename, m_type in models_to_check.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            report["load_tests"][filename] = "❌ MISSING FILE"
            continue
        try:
            if m_type == "xgb":
                xgb.Booster(model_file=path)
            elif m_type == "lgb":
                lgb.Booster(model_file=path)
            elif m_type == "cat":
                c = CatBoostClassifier()
                c.load_model(path)
            report["load_tests"][filename] = "✅ LOADED OK"
        except Exception as e:
            report["load_tests"][filename] = f"❌ ERROR: {str(e)}"

    return JsonResponse(report, json_dumps_params={'indent': 2})


@login_required
def ops_dashboard_view(request):
    # SECURITY: Lock this to Superusers only
    if not request.user.is_superuser:
        return redirect('terminal')

    from django.contrib.auth.models import User
    from .models import JournalEntry, UserProfile
    from django.db.models import Sum

    # 1. High Level Metrics
    total_users = User.objects.count()
    active_subs = UserProfile.objects.filter(is_premium=True).count()

    # 2. PnL Stats (All Users)
    total_signals = JournalEntry.objects.count()
    wins = JournalEntry.objects.filter(status='WIN').count()
    win_rate = round((wins / total_signals * 100), 1) if total_signals > 0 else 0

    # 3. Recent Signals Feed
    recent_signals = JournalEntry.objects.select_related('user').order_by('-created_at')[:20]

    context = {
        'total_users': total_users,
        'active_subs': active_subs,
        'total_signals': total_signals,
        'win_rate': win_rate,
        'recent_signals': recent_signals
    }
    return render(request, 'core/ops_dashboard.html', context)