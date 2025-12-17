import razorpay
import os
import json
from datetime import datetime

from django.core.paginator import Paginator
from django.http import JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone
from datetime import timedelta
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile, JournalEntry
from .forms import SignupForm, UserUpdateForm
from .services.marketdata_service import MarketService
from django.core.mail import send_mail
from django.conf import settings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
import requests

from .utils import analyze_market_data


# --- PUBLIC PAGES ---
def landing_view(request):
    if request.user.is_authenticated:
        return redirect('terminal')
    return render(request, 'core/landing.html')


# --- AUTHENTICATION ---
def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            # 1. CRITICAL: Save User & Trigger Trial Logic
            # This line creates the User. The signal in models.py immediately
            # creates the UserProfile and sets 'trial_start_date'.
            # Trial logic is now SECURE.
            user = form.save()

            # 2. EMAIL LOGIC (Safe Mode)
            # We wrap this in try/except so it NEVER blocks the signup process.
            try:
                subject = "Access Granted: Reelioo Neural Terminal Online"

                # The HTML Content
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

                # Plain text fallback for old email clients
                plain_message = strip_tags(html_message)

                # Send Welcome Email
                send_mail(
                    subject,
                    plain_message,
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    html_message=html_message,
                    fail_silently=True,  # Keeps app running if email fails
                )

                # Send Admin Alert to You
                send_mail(
                    f"🚀 New Signup: {user.username}",
                    f"Email: {user.email}\nCountry: {user.profile.country}",
                    settings.DEFAULT_FROM_EMAIL,
                    ['reeliooapp@gmail.com'],  # Your admin email
                    fail_silently=True,
                )

            except Exception as e:
                # Log error but DO NOT stop the user from logging in
                print(f"⚠️ Email System Error: {e}")

            # 3. Log In & Redirect
            # This ensures the user enters the app immediately
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
        except User.DoesNotExist:
            messages.error(request, "No account found.")

    return render(request, 'core/auth/login.html')


def logout_view(request):
    logout(request)
    return redirect('landing')



# --- TERMINAL ---
@login_required(login_url='login')
def terminal_view(request):
    profile = request.user.profile

    # This runs the "Lazy Check" inside the model
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

    # 1. BLOCK UPGRADE if user is 'cancellation_pending' (Still has access)
    if profile.subscription_status == 'cancellation_pending':
        if profile.is_access_granted():
            messages.info(request, "You have an active plan. Wait for it to expire before resubscribing.")
            return redirect('settings')

    # 2. STANDARD LOGIC
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

                # UPDATE LOGIC:
                profile.is_premium = True
                profile.subscription_status = "active"

                # Set approximate end date (30 days from now)
                # Ideally, webhooks handle renewal, but this ensures immediate access
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


# --- CANCEL SUBSCRIPTION (LOGIC FIX) ---
@login_required
def cancel_subscription_view(request):
    if request.method == "POST":
        profile = request.user.profile
        sub_id = profile.razorpay_subscription_id

        if sub_id and profile.is_premium:
            try:
                client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

                # 1. Fetch Subscription Data FIRST to get the "current_end"
                sub_details = client.subscription.fetch(sub_id)

                # Razorpay returns 'current_end' as a Unix timestamp
                current_end_timestamp = sub_details.get('current_end')

                # Convert to Django Datetime
                if current_end_timestamp:
                    end_date = datetime.fromtimestamp(current_end_timestamp)
                    # Make it timezone aware
                    end_date = timezone.make_aware(end_date)
                    profile.subscription_end_date = end_date
                else:
                    # Fallback if API fails: Keep current date + 30 days or existing date
                    if not profile.subscription_end_date:
                        profile.subscription_end_date = timezone.now() + timedelta(days=30)

                # 2. Cancel at Cycle End (User keeps access until paid period is over)
                client.subscription.cancel(sub_id, {'cancel_at_cycle_end': 1})

                # 3. Update Local DB
                # DO NOT set is_premium = False yet!
                profile.subscription_status = "cancellation_pending"
                profile.save()

                return render(request, 'core/cancel_success.html')

            except Exception as e:
                print(f"Cancel Error: {e}")
                messages.error(request, "Could not cancel. Contact support.")

    return redirect('settings')


# ... (Settings view remains mostly same, just Logic checks in template) ...
@login_required
def settings_view(request):
    # Same code as provided previously
    user = request.user
    profile = user.profile

    # Run lazy check ensures data is fresh
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


@login_required
def journal_view(request):
    # Fetch user's entries
    entries_list = JournalEntry.objects.filter(user=request.user)

    # Pagination (10 per page)
    paginator = Paginator(entries_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Calculate Stats
    total_trades = entries_list.count()
    wins = entries_list.filter(status='WIN').count()
    losses = entries_list.filter(status='LOSS').count()
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
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Create Entry
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
    if request.method == "DELETE":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)
            entry.delete()
            return JsonResponse({'status': 'success'})
        except JournalEntry.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not found'}, status=404)


@login_required
def refresh_journal_entry(request, entry_id):
    if request.method == "POST":
        try:
            entry = JournalEntry.objects.get(id=entry_id, user=request.user)

            # 1. Fetch Live Price using your MarketService
            # Note: Ensure MarketService.get_historical_data is working correctly
            df = MarketService.get_historical_data(entry.symbol, "PERP", "SCALP")

            if df is None or df.empty:
                return JsonResponse({'status': 'error', 'message': 'Market data unavailable'})

            current_price = float(df['close'].iloc[-1])

            # 2. Compare Price vs Targets
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

            # 3. Save Changes
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


# --- CRON JOB TRIGGER (WEBHOOK) ---
def cron_scan_trigger(request, secret_key):
    # 1. Security Check via Settings
    required_secret = getattr(settings, 'CRON_SECRET', 'super-secret-password-123')
    if secret_key != required_secret:
        return JsonResponse({'status': 'forbidden', 'message': 'Access Denied'}, status=403)

    # 2. Define Assets
    watchlist = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    alerts_sent = 0
    scanned = 0

    try:
        for symbol in watchlist:
            # Run AI Logic (Imported from utils)
            data = analyze_market_data(symbol)

            # Check if valid data returned
            if not data or 'signal' not in data:
                continue

            score = data.get('signal', {}).get('probability', 0)
            bias = data.get('signal', {}).get('bias', 'NEUTRAL')

            # 3. SNIPER FILTER (>65%)
            if score >= 65 and bias != 'NEUTRAL':
                send_discord_alert(data)
                alerts_sent += 1
            scanned += 1

        return JsonResponse({
            'status': 'success',
            'scanned': scanned,
            'alerts_sent': alerts_sent
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def send_discord_alert(data):
    # Fetch URL from Env
    webhook_url = os.getenv('DISCORD_URL')

    if not webhook_url:
        print("Error: DISCORD_URL not found in .env")
        return

    # Logic to pick color (Green for Long, Red for Short)
    color = 5763719 if data['signal']['bias'] == 'LONG' else 15548997

    payload = {
        "username": "Reelioo Sniper Bot",
        "avatar_url": "https://i.imgur.com/6Xy1sJ2.png",  # Generic bot icon
        "embeds": [{
            "title": f"🚨 SNIPER SIGNAL: {data['symbol']}",
            "description": f"**High Conviction Setup Detected.**\nThe AI has identified a precision entry zone.",
            "color": color,
            "fields": [
                {"name": "Bias", "value": f"**{data['signal']['bias']}**", "inline": True},
                {"name": "Confidence", "value": f"**{data['signal']['probability']}%**", "inline": True},
                {"name": "Entry Zone", "value": f"`${data['signal']['entry']}`", "inline": True},
                {"name": "Stop Loss", "value": f"${data['signal']['stop']}", "inline": True},
                {"name": "Target", "value": f"${data['signal']['target2']}", "inline": True}
            ],
            "footer": {"text": "Reelioo Institutional Terminal • Time: Live"}
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


