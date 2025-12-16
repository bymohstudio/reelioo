from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from django.db.models.signals import post_save
from django.dispatch import receiver
from datetime import datetime


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    # Subscription Logic
    trial_start_date = models.DateTimeField(auto_now_add=True)

    # We keep is_premium as a flag, but access depends on the End Date now
    is_premium = models.BooleanField(default=False)

    # NEW: Stores exactly when the paid access ends
    subscription_end_date = models.DateTimeField(blank=True, null=True)

    razorpay_subscription_id = models.CharField(max_length=100, blank=True, null=True)

    # Statuses: 'trial', 'active', 'cancellation_pending', 'expired'
    subscription_status = models.CharField(max_length=50, default="trial")

    # Extended Profile Fields
    country = models.CharField(max_length=100, blank=True, null=True)
    terms_accepted = models.BooleanField(default=False)

    def is_access_granted(self):
        """
        Master check for Terminal Access.
        """
        # 1. Check Paid Access (Active OR Cancellation Pending but time remains)
        if self.is_premium:
            # Lazy Expiration Check: If end date exists and we are PAST it
            if self.subscription_end_date and timezone.now() > self.subscription_end_date:
                self.perform_lazy_expiration()  # Revoke access
                return False
            return True

        # 2. Check Trial Access
        if not self.is_trial_expired():
            return True

        return False

    def perform_lazy_expiration(self):
        """Helper to downgrade user if time is up"""
        if self.is_premium:
            self.is_premium = False
            self.subscription_status = "expired"
            self.save()

    def is_trial_expired(self):
        trial_end = self.trial_start_date + timedelta(days=21)
        return timezone.now() > trial_end

    def get_days_left(self):
        # Case A: Premium (Active or Cancelling)
        if self.is_premium and self.subscription_end_date:
            remaining = self.subscription_end_date - timezone.now()
            days = max(0, remaining.days)
            return f"{days} DAYS (PREMIUM)"

        # Case B: Lifetime/Manual Premium (No date set)
        if self.is_premium:
            return "UNLIMITED"

        # Case C: Trial
        trial_end = self.trial_start_date + timedelta(days=21)
        remaining = trial_end - timezone.now()
        return max(0, remaining.days)

    def __str__(self):
        return f"{self.user.username} | Status: {self.subscription_status}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)



# --- ANALYTICS MODELS (For future Admin Dashboard) ---
class PredictionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=20)
    score = models.FloatField()
    bias = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.symbol} - {self.bias} ({self.score})"