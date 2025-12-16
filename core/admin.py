from django.contrib import admin
from .models import UserProfile, JournalEntry, PredictionLog

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_email', 'country', 'subscription_status', 'is_premium', 'get_trial_status')
    list_filter = ('is_premium', 'subscription_status', 'country', 'trial_start_date')
    search_fields = ('user__username', 'user__email', 'country', 'razorpay_subscription_id')
    readonly_fields = ('trial_start_date',)
    ordering = ('-trial_start_date',)

    # Helper to show Email from the related User model
    def get_email(self, obj):
        return obj.user.email
    get_email.short_description = 'Email'

    # Helper to show live days remaining
    def get_trial_status(self, obj):
        return obj.get_days_left()
    get_trial_status.short_description = 'Access / Days Left'

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ('user', 'symbol', 'bias', 'status', 'pnl_percent', 'confidence', 'created_at')
    list_filter = ('status', 'bias', 'created_at')
    search_fields = ('user__username', 'user__email', 'symbol')
    ordering = ('-created_at',)

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'bias', 'score', 'created_at')
    list_filter = ('bias', 'created_at')
    search_fields = ('symbol',)
    ordering = ('-created_at',)
