from django.contrib import admin
from .models import UserProfile, JournalEntry

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    # Match the new Gumroad fields
    list_display = (
        'user',
        'gumroad_license_key',
        'is_premium',
        'last_verified_at',
        'terms_accepted'
    )

    # Filter by Premium status
    list_filter = ('is_premium', 'terms_accepted')

    # Search by Username or License Key
    search_fields = ('user__username', 'user__email', 'gumroad_license_key')

    # Edit Form Layout
    fieldsets = (
        ('User Info', {
            'fields': ('user', 'terms_accepted')
        }),
        ('Gumroad Access', {
            'fields': ('gumroad_license_key', 'is_premium', 'last_verified_at')
        }),
    )

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    # This remains the same
    list_display = ('user', 'symbol', 'bias', 'status', 'pnl_percent', 'created_at')
    list_filter = ('status', 'bias', 'created_at')
    search_fields = ('symbol', 'user__username')