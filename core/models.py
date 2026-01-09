from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    # --- LEMON SQUEEZY SYNC ---
    lemon_squeezy_customer_id = models.CharField(max_length=100, blank=True, null=True)
    lemon_squeezy_subscription_id = models.CharField(max_length=100, blank=True, null=True)

    # Status: 'active', 'on_trial', 'past_due', 'cancelled', 'expired'
    # Default is 'inactive' because new signups don't have a sub yet.
    subscription_status = models.CharField(max_length=50, default="inactive")

    # When access truly ends (synced from LS 'renews_at' or 'ends_at')
    renews_at = models.DateTimeField(blank=True, null=True)

    # URL to update card details (synced from LS webhook)
    update_payment_url = models.URLField(blank=True, null=True)

    # Meta
    country = models.CharField(max_length=100, blank=True, null=True)
    terms_accepted = models.BooleanField(default=False)

    @property
    def is_premium(self):
        """
        Determines if the user gets Pro Access.
        PRIORITY 1: GOD MODE (Superuser)
        PRIORITY 2: Lemon Squeezy Status
        """
        # 1. GOD MODE: You always have access
        if self.user.is_superuser:
            return True

        # 2. Valid Paid/Trial Statuses
        # 'on_trial' means they are in the 14-day LS trial.
        # 'active' means they are paying.
        valid_statuses = ['active', 'on_trial']
        if self.subscription_status in valid_statuses:
            return True

        # 3. Grace Period (Cancelled but not expired yet)
        if self.subscription_status == 'cancelled' and self.renews_at:
            return timezone.now() < self.renews_at

        return False

    def __str__(self):
        return f"{self.user.username} | {self.subscription_status}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


# --- JOURNAL & ANALYTICS (Unchanged) ---
class JournalEntry(models.Model):
    STATUS_CHOICES = [('PENDING', 'Pending'), ('WIN', 'Win'), ('LOSS', 'Loss'), ('BREAKEVEN', 'Breakeven')]
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="journal_entries")
    symbol = models.CharField(max_length=20)
    bias = models.CharField(max_length=10)
    entry_price = models.FloatField()
    stop_loss = models.FloatField()
    target = models.FloatField()
    confidence = models.FloatField(default=0.0)
    leverage = models.CharField(max_length=20, default="Low")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    pnl_percent = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta: ordering = ['-created_at']

    def __str__(self): return f"{self.symbol} ({self.bias}) - {self.status}"


class PredictionLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=20)
    score = models.FloatField()
    bias = models.CharField(max_length=10)

    def __str__(self): return f"{self.symbol} - {self.bias} ({self.score})"