from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    # --- GUMROAD AUTH ---
    gumroad_license_key = models.CharField(max_length=100, unique=True, null=True, blank=True)

    # Status
    is_premium = models.BooleanField(default=False)
    last_verified_at = models.DateTimeField(null=True, blank=True)

    # Meta (Optional now, but good to keep)
    terms_accepted = models.BooleanField(default=True)

    def needs_verification(self):
        """Check status if we haven't checked in 24 hours"""
        if not self.last_verified_at:
            return True
        return (timezone.now() - self.last_verified_at).days >= 1

    def is_access_granted(self):
        if self.user.is_superuser: return True
        return self.is_premium

    def __str__(self):
        return f"{self.user.username} | {self.gumroad_license_key}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


# --- JOURNAL ENTRY (Keep exactly as is) ---
class JournalEntry(models.Model):
    # ... (Paste your existing JournalEntry code here, no changes needed) ...
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

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.symbol} ({self.bias}) - {self.status}"