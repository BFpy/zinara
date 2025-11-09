from django.contrib import admin
from .models import DashboardPreferences, Alert

@admin.register(DashboardPreferences)
class DashboardPreferencesAdmin(admin.ModelAdmin):
    list_display = ['user', 'notifications_enabled', 'email_alerts']

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ['title', 'alert_type', 'severity', 'vehicle_id', 'created_at', 'acknowledged']
    list_filter = ['alert_type', 'severity', 'acknowledged', 'created_at']
    search_fields = ['title', 'vehicle_id']
