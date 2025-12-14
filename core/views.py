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
    RAZORPAY_PLAN_ID = os.getenv("RAZORPAY_PLAN_ID", "plan_placeholder_id")
    client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

    # Conditional Context Logic
    profile = request.user.profile
    is_expired = profile.is_trial_expired() and not profile.is_premium

    context = {
        "key_id": os.getenv("RAZORPAY_KEY_ID"),
        "user_email": request.user.email,
        "is_trial_expired": is_expired,  # <-- Pass this flag to template
        # Subscription creation logic can go here or be dynamic via JS API
    }
    return render(request, 'core/pricing.html', context)


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

# --- SUBSCRIPTION (MONTHLY) ---
@login_required
def pricing_view(request):
    # Razorpay Monthly Plan ID (Create this in Razorpay Dashboard)
    # Price: $19 (approx 1600 INR)
    RAZORPAY_PLAN_ID = os.getenv("RAZORPAY_PLAN_ID", "plan_placeholder_id")

    client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

    # Create Subscription Object
    try:
        subscription = client.subscription.create({
            "plan_id": RAZORPAY_PLAN_ID,
            "total_count": 60, # 5 Years
            "quantity": 1,
            "customer_notify": 1,
            "notes": {
                "user_id": request.user.id,
                "email": request.user.email
            }
        })
        sub_id = subscription['id']
    except Exception as e:
        sub_id = "error_creating_sub"
        print(f"Razorpay Error: {e}")

    context = {
        "sub_id": sub_id,
        "key_id": os.getenv("RAZORPAY_KEY_ID"),
        "user_email": request.user.email
    }
    return render(request, 'core/pricing.html', context)

@csrf_exempt
@login_required
def payment_success(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            # In production, verify signature here using client.utility.verify_payment_signature

            profile = request.user.profile
            profile.is_premium = True
            profile.subscription_status = "active"
            profile.razorpay_subscription_id = data.get('razorpay_subscription_id')
            profile.save()

            return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "invalid_method"})

# --- LEGAL PAGES ---
def terms_view(request): return render(request, 'core/legal/terms.html')
def privacy_view(request): return render(request, 'core/legal/privacy.html')
def refund_view(request): return render(request, 'core/legal/refund.html')
def contact_view(request): return render(request, 'core/legal/contact.html')
def pricing_footer_view(request): return render(request, 'core/legal/pricing_footer.html')


