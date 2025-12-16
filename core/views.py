import razorpay
import os
import json
from datetime import datetime
from django.utils import timezone
from datetime import timedelta
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile
from .forms import SignupForm, UserUpdateForm


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
            user = form.save()
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



# --- LEGAL PAGES ---
def terms_view(request): return render(request, 'core/legal/terms.html')
def privacy_view(request): return render(request, 'core/legal/privacy.html')
def refund_view(request): return render(request, 'core/legal/refund.html')
def contact_view(request): return render(request, 'core/legal/contact.html')
def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


