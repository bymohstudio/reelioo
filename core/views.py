import razorpay
import os
import json
import requests
import logging
from datetime import datetime, timedelta

from django.core.paginator import Paginator
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
# --- SAFE IMPORTS (Services/Utils usually safe) ---
from .services.marketdata_service import MarketService
from core.utils import analyze_market_data

# Add these imports at the top of views.py if missing
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# NOTE: 'models' and 'forms' are imported inside functions to prevent Circular Import Recursion

# --- PUBLIC PAGES ---
def landing_view(request):
    if request.user.is_authenticated:
        return redirect('terminal')
    return render(request, 'core/landing.html')


# --- AUTHENTICATION ---
def signup_view(request):
    # LAZY IMPORT to stop RecursionError
    from .forms import SignupForm

    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()

            # EMAIL LOGIC (Safe Mode)
            try:
                subject = "Access Granted: Reelioo Neural Terminal Online"
                html_message = f"""
                <!DOCTYPE html>
                <html>
                <body style="background-color: #000; font-family: sans-serif; color: #ccc;">
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
                </body>
                </html>
                """
                plain_message = strip_tags(html_message)

                send_mail(
                    subject,
                    plain_message,
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    html_message=html_message,
                    fail_silently=True,
                )

                # Admin Alert
                send_mail(
                    f"🚀 New Signup: {user.username}",
                    f"Email: {user.email}",
                    settings.DEFAULT_FROM_EMAIL,
                    ['reeliooapp@gmail.com'],
                    fail_silently=True,
                )

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

    context = {
        'days_left': profile.get_days_left(),
        'is_premium': profile.is_premium,
        'username': request.user.username
    }
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
            subscription = client.subscription.create({
                "plan_id": plan_id,
                "total_count": 60,
                "quantity": 1,
                "customer_notify": 1,
                "notes": {"email": request.user.email}
            })
            sub_id = subscription['id']
            profile.razorpay_subscription_id = sub_id
            profile.save()

    except Exception as e:
        print(f"Razorpay Init Error: {e}")

    context = {
        "key_id": key_id,
        "sub_id": sub_id,
        "user_email": request.user.email,
        "is_trial_expired": profile.is_trial_expired() and not profile.is_premium
    }
    return render(request, 'core/pricing.html', context)


# --- PAYMENT SUCCESS ---
@csrf_exempt
def payment_success_view(request):
    from .models import UserProfile  # Lazy Import

    if request.method == "POST":
        try:
            payment_id = request.POST.get('razorpay_payment_id')
            subscription_id = request.POST.get('razorpay_subscription_id')
            signature = request.POST.get('razorpay_signature')

            client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
            data_to_verify = {
                'razorpay_payment_id': payment_id,
                'razorpay_subscription_id': subscription_id,
                'razorpay_signature': signature
            }
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
    from .forms import UserUpdateForm  # Lazy Import

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
    from .models import JournalEntry  # Lazy Import

    entries_list = JournalEntry.objects.filter(user=request.user)
    paginator = Paginator(entries_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    total_trades = entries_list.count()
    wins = entries_list.filter(status='WIN').count()
    win_rate = round((wins / total_trades * 100), 1) if total_trades > 0 else 0

    context = {
        'page_obj': page_obj,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'active_pending': entries_list.filter(status='PENDING').count()
    }
    return render(request, 'core/journal.html', context)


@login_required
def add_journal_entry(request):
    from .models import JournalEntry  # Lazy Import

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            JournalEntry.objects.create(
                user=request.user,
                symbol=data.get('symbol'),
                bias=data.get('bias'),
                entry_price=float(data.get('entry')),
                stop_loss=float(data.get('stop')),
                target=float(data.get('target')),
                confidence=float(data.get('confidence', 0)),
                leverage=data.get('leverage', 'Low')
            )
            return JsonResponse({'status': 'success', 'message': 'Signal saved to Watchlist'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
def delete_journal_entry(request, entry_id):
    from .models import JournalEntry  # Lazy Import

    if request.method == "DELETE":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)
            entry.delete()
            return JsonResponse({'status': 'success'})
        except JournalEntry.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not found'}, status=404)


@login_required
def refresh_journal_entry(request, entry_id):
    from .models import JournalEntry  # Lazy Import

    if request.method == "POST":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)

            # 1. STRICT 24-HOUR RULE (Institutional Velocity)
            # If a trade is pending > 24h, it is dead capital. Close it.
            now = timezone.now()
            duration = now - entry.created_at
            hours_open = duration.total_seconds() / 3600

            MAX_DURATION = 24  # STRICT LIMIT

            if entry.status == "PENDING" and hours_open > MAX_DURATION:
                entry.status = "TIMEOUT"
                entry.save()
                return JsonResponse({
                    'status': 'success',
                    'new_status': "TIMEOUT",
                    'message': f"Trade Closed: Velocity Limit Exceeded ({int(hours_open)}h). Dead capital released."
                })

            # 2. Check Price Targets
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")
            if df is None or df.empty:
                return JsonResponse({'status': 'error', 'message': 'Market data unavailable'})

            current_price = float(df['close'].iloc[-1])
            new_status = "PENDING"

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

            if new_status != "PENDING":
                entry.status = new_status
                entry.save()

            return JsonResponse({
                'status': 'success',
                'new_status': new_status,
                'current_price': current_price
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid Method'}, status=405)


def robots_view(request):
    content = render_to_string('core/robots.txt')
    return HttpResponse(content, content_type="text/plain")


def sitemap_view(request):
    content = render_to_string('core/sitemap.xml')
    return HttpResponse(content, content_type="application/xml")


# --- CRON JOB TRIGGER ---
def cron_scan_trigger(request, secret_key):
    required_secret = getattr(settings, 'CRON_SECRET', 'super-secret-password-123')
    if secret_key != required_secret:
        return JsonResponse({'status': 'forbidden', 'message': 'Access Denied'}, status=403)

    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    alerts_sent = 0
    scanned = 0
    logs = []

    try:
        engine = CryptoQuantEngine()

        for symbol in watchlist:
            df = MarketService.get_historical_data(symbol, "AUTO", "INTRADAY")
            if df is None or df.empty: continue

            res = engine.analyze(df, "INTRADAY")

            # --- TIERED LOGIC ---

            # TIER 1: SNIPER SIGNAL (The 65% "Kill Shot")
            if res.score >= 65 and res.bias != 'NEUTRAL':
                send_discord_alert(symbol, res, alert_type="SNIPER")
                alerts_sent += 1
                logs.append(f"{symbol}: SNIPER SENT ({res.score}%)")

            # TIER 2: WATCH SIGNAL (The 60-64% "Heads Up")
            # We alert ONLY if it's "Active" (60%) but not yet "Sniper" (65%)
            elif res.score >= 60 and res.bias != 'NEUTRAL':
                send_discord_alert(symbol, res, alert_type="WATCH")
                alerts_sent += 1
                logs.append(f"{symbol}: WATCH SENT ({res.score}%)")

            else:
                logs.append(f"{symbol}: Low Score ({res.score}%)")

            scanned += 1

        return JsonResponse({'status': 'success', 'scanned': scanned, 'alerts_sent': alerts_sent, 'logs': logs})

    except Exception as e:
        import traceback
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# UPDATED DISCORD FUNCTION TO HANDLE TIERS
def send_discord_alert(symbol, data, alert_type="SNIPER"):
    webhook_url = os.getenv('DISCORD_URL')
    if not webhook_url: return

    # COLORS
    if alert_type == "SNIPER":
        # Green for Long, Red for Short
        color = 5763719 if data.bias == 'LONG' else 15548997
        title = f"🚨 SNIPER SIGNAL: {symbol}"
        desc = "**HIGH CONVICTION SETUP CONFIRMED.**"
    else:
        # Yellow for Watch
        color = 16776960
        title = f"👀 WATCH ALERT: {symbol}"
        desc = "*Bias detected. Waiting for volatility confirmation.*"


    avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"

    payload = {
        "username": "Reelioo Terminal Bot",
        "avatar_url": avatar,
        "embeds": [{
            "title": title,
            "description": desc,
            "color": color,
            "fields": [
                {"name": "Bias", "value": f"**{data.bias}**", "inline": True},
                {"name": "Confidence", "value": f"**{data.score}%**", "inline": True},
                {"name": "Price", "value": f"`${data.entry}`", "inline": True},
                {"name": "Stop", "value": f"${data.stop}", "inline": True},
                {"name": "Target", "value": f"${data.target2}", "inline": True}
            ],
            "footer": {"text": "Reelioo Institutional Terminal"}
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
    """
    Diagnostic view to find why models aren't loading.
    Access at: /debug-models/
    """
    # 1. Calculate Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quant_dir = os.path.join(base_dir, "quant")
    model_dir = os.path.join(quant_dir, "ml_models")

    report = {
        "scanned_path": model_dir,
        "exists": os.path.exists(model_dir),
        "files_found": [],
        "load_tests": {}
    }

    # 2. List Files & Check Sizes
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        for f in files:
            f_path = os.path.join(model_dir, f)
            size_kb = os.path.getsize(f_path) / 1024
            report["files_found"].append(f"{f} ({size_kb:.2f} KB)")

            # Warn if file is suspiciously small (LFS Pointer check)
            if size_kb < 2.0:
                report["load_tests"][f] = "⚠️ WARNING: File too small. Likely Git LFS pointer, not real model."

    # 3. Try Loading (Catch exact error)
    # Define expected files
    models_to_check = {
        "xgb_long.json": "xgb",
        "xgb_short.json": "xgb",
        "lgb_long.txt": "lgb",
        "lgb_short.txt": "lgb",
        "cat_long.cbm": "cat",
        "cat_short.cbm": "cat"
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