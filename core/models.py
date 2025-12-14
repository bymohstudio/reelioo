from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    # Subscription Logic
    trial_start_date = models.DateTimeField(auto_now_add=True)
    is_premium = models.BooleanField(default=False)
    razorpay_subscription_id = models.CharField(max_length=100, blank=True, null=True)
    subscription_status = models.CharField(max_length=20, default="trial")

    # Extended Profile Fields
    country = models.CharField(max_length=100, blank=True, null=True)
    terms_accepted = models.BooleanField(default=False)

    def is_access_granted(self):
        # 1. Premium Check
        if self.is_premium and self.subscription_status == "active":
            return True

        # 2. Trial Check
        if not self.is_trial_expired():
            return True

        return False

    def is_trial_expired(self):
        trial_end = self.trial_start_date + timedelta(days=21)
        return timezone.now() > trial_end

    def get_days_left(self):
        if self.is_premium: return "LIFETIME"
        trial_end = self.trial_start_date + timedelta(days=21)
        remaining = trial_end - timezone.now()
        return max(0, remaining.days)

    def __str__(self):
        return f"{self.user.username} | Premium: {self.is_premium}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


# --- AUTO-CREATE PROFILE SIGNAL ---
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


# --- ANALYTICS MODELS (For future Admin Dashboard) ---
class PredictionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=20)
    score = models.FloatField()
    bias = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.symbol} - {self.bias} ({self.score})"