import razorpay
import os
import json
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile
from .forms import SignupForm, UserUpdateForm  # Import the new forms


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
    if not profile.is_access_granted():
        messages.warning(request, "Trial Expired. Please Upgrade.")
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
    # 1. Fetch Keys & Config
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    plan_id = os.getenv("RAZORPAY_PLAN_ID")

    client = razorpay.Client(auth=(key_id, key_secret))
    profile = request.user.profile

    # 2. Create Razorpay Subscription
    sub_id = "error"
    try:
        if key_id and key_secret and plan_id:
            subscription = client.subscription.create({
                "plan_id": plan_id,
                "total_count": 60,  # 5 Years
                "quantity": 1,
                "customer_notify": 1,
                "notes": {"email": request.user.email}
            })
            sub_id = subscription['id']

            # 🚀 PRE-SAVE: Link this Sub ID to the User NOW
            # This allows us to find them if the session drops during payment
            profile.razorpay_subscription_id = sub_id
            profile.save()

    except Exception as e:
        print(f"Razorpay Init Error: {e}")

    # 3. Context for Template
    context = {
        "key_id": key_id,
        "sub_id": sub_id,
        "user_email": request.user.email,
        "is_trial_expired": profile.is_trial_expired() and not profile.is_premium
    }
    return render(request, 'core/pricing.html', context)


# --- PAYMENT SUCCESS HANDLER ---
@csrf_exempt
def payment_success_view(request):
    if request.method == "POST":
        try:
            # 1. Capture Data
            payment_id = request.POST.get('razorpay_payment_id')
            subscription_id = request.POST.get('razorpay_subscription_id')
            signature = request.POST.get('razorpay_signature')

            # 2. Verify Signature
            client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
            data_to_verify = {
                'razorpay_payment_id': payment_id,
                'razorpay_subscription_id': subscription_id,
                'razorpay_signature': signature
            }
            client.utility.verify_subscription_payment_signature(data_to_verify)

            # 3. 🚀 RECOVER USER (Fixes "Logged Out" Issue)
            try:
                # Find the profile that has this subscription ID
                profile = UserProfile.objects.get(razorpay_subscription_id=subscription_id)
                user = profile.user

                # 4. Activate Premium
                profile.is_premium = True
                profile.subscription_status = "active"
                profile.save()

                # 5. 🚀 FORCE RE-LOGIN
                # Manually restore the session so they are authenticated in the popup
                user.backend = 'django.contrib.auth.backends.ModelBackend'
                login(request, user)

                # Render the Beautiful Success Popup
                return render(request, 'core/success.html')

            except UserProfile.DoesNotExist:
                print("❌ Fatal: Payment success but UserProfile not found.")
                return redirect('pricing')

        except Exception as e:
            print(f"❌ Verification Failed: {e}")
            return render(request, 'core/payment_failed.html')

    return redirect('pricing')


# ... existing imports ...

@login_required
def cancel_subscription_view(request):
    if request.method == "POST":
        profile = request.user.profile
        sub_id = profile.razorpay_subscription_id

        if sub_id and profile.is_premium:
            try:
                # 1. Initialize Razorpay
                client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

                # 2. Cancel on Razorpay (cancel_at_cycle_end=0 implies immediate cancellation)
                client.subscription.cancel(sub_id, {'cancel_at_cycle_end': 0})

                # 3. Update Local DB
                profile.is_premium = False
                profile.subscription_status = "cancelled"
                profile.save()

                return render(request, 'core/cancel_success.html')

            except Exception as e:
                print(f"Cancel Error: {e}")
                messages.error(request, "Could not cancel subscription. Please try again or contact support.")

    return redirect('settings')

# --- ACCOUNT SETTINGS (NEW) ---
@login_required
def settings_view(request):
    user = request.user
    profile = user.profile

    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=user)
        if form.is_valid():
            user = form.save(commit=False)
            # Handle manual country update since it's not on User model
            profile.country = form.cleaned_data.get('country', profile.country)
            user.save()
            profile.save()
            messages.success(request, "Profile Updated Successfully.")
            return redirect('settings')
    else:
        # Pre-fill form
        initial_data = {'country': profile.country}
        form = UserUpdateForm(instance=user, initial=initial_data)

    return render(request, 'core/auth/settings.html', {'form': form})



# --- LEGAL PAGES ---
def terms_view(request): return render(request, 'core/legal/terms.html')
def privacy_view(request): return render(request, 'core/legal/privacy.html')
def refund_view(request): return render(request, 'core/legal/refund.html')
def contact_view(request): return render(request, 'core/legal/contact.html')
def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


