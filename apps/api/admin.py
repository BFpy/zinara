from django.contrib import admin
from .models import APIKey, APIRequestLog

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['user', 'key', 'is_active', 'created_at', 'last_used']
    list_filter = ['is_active', 'created_at']
    readonly_fields = ['key', 'created_at', 'last_used']

@admin.register(APIRequestLog)
class APIRequestLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'endpoint', 'method', 'status_code', 'response_time', 'timestamp']
    list_filter = ['method', 'status_code', 'timestamp']
    readonly_fields = ['timestamp']
    search_fields = ['endpoint', 'user__username']
