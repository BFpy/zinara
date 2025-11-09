from django.contrib import admin
from .models import VehicleRecord, PredictionLog, DataUpload

@admin.register(VehicleRecord)
class VehicleRecordAdmin(admin.ModelAdmin):
    list_display = ['vehicle_id', 'vehicle_type', 'region', 'is_currently_licensed', 
                    'last_license_renewal', 'risk_score', 'predicted_unlicensed']
    list_filter = ['vehicle_type', 'region', 'is_currently_licensed', 'predicted_unlicensed']
    search_fields = ['vehicle_id']
    readonly_fields = ['created_at', 'updated_at', 'last_prediction_date']
    
    fieldsets = (
        ('Vehicle Information', {
            'fields': ('vehicle_id', 'vehicle_type', 'registration_date', 'region')
        }),
        ('Licensing Status', {
            'fields': ('is_currently_licensed', 'last_license_renewal', 'license_expiry_date')
        }),
        ('Payment Information', {
            'fields': ('preferred_payment_mode', 'total_renewals', 'late_renewals_count', 'average_renewal_delay')
        }),
        ('Agent Information', {
            'fields': ('last_agent_sync', 'agent_sync_delay')
        }),
        ('ML Predictions', {
            'fields': ('risk_score', 'predicted_unlicensed', 'last_prediction_date'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ['vehicle', 'prediction_date', 'risk_score', 'predicted_unlicensed', 'model_version']
    list_filter = ['predicted_unlicensed', 'model_version', 'prediction_date']
    readonly_fields = ['prediction_date']

@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ['file_name', 'uploaded_by', 'upload_date', 'processed', 'records_count']
    list_filter = ['processed', 'upload_date']
    readonly_fields = ['upload_date']
