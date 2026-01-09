from django.contrib import admin
from .models import UserProfile, JournalEntry


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    # Display these columns in the list view
    list_display = (
        'user',
        'subscription_status',
        'is_premium_display',  # Custom method to show the property
        'renews_at',
        'country'
    )

    # Filter sidebar (Must be DB fields)
    list_filter = ('subscription_status', 'terms_accepted')

    # Search bar
    search_fields = ('user__username', 'user__email', 'lemon_squeezy_customer_id')

    # Fields to show in the edit form
    fieldsets = (
        ('User Info', {
            'fields': ('user', 'country', 'terms_accepted')
        }),
        ('Lemon Squeezy Sync', {
            'fields': ('subscription_status', 'lemon_squeezy_customer_id', 'lemon_squeezy_subscription_id', 'renews_at',
                       'update_payment_url')
        }),
    )

    # Make technical IDs read-only so you don't accidentally break sync
    readonly_fields = ('lemon_squeezy_customer_id', 'lemon_squeezy_subscription_id', 'update_payment_url')

    # Helper to display the @property 'is_premium' in the admin list
    def is_premium_display(self, obj):
        return obj.is_premium

    is_premium_display.boolean = True  # Shows a Green Check / Red X icon
    is_premium_display.short_description = "Premium Access"


@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ('user', 'symbol', 'bias', 'status', 'pnl_percent', 'created_at')
    list_filter = ('status', 'bias', 'created_at')
    search_fields = ('symbol', 'user__username')