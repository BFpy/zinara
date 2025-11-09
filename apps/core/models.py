from django.db import models
from django.contrib.auth.models import User

class VehicleRecord(models.Model):
    VEHICLE_TYPES = [
        ('sedan', 'Sedan'),
        ('suv', 'SUV'),
        ('truck', 'Truck'),
        ('motorcycle', 'Motorcycle'),
        ('bus', 'Bus'),
        ('other', 'Other'),
    ]
    
    REGIONS = [
        ('urban', 'Urban'),
        ('peri_urban', 'Peri-Urban'),
        ('rural', 'Rural'),
    ]
    
    PAYMENT_MODES = [
        ('online', 'Online'),
        ('mobile_money', 'Mobile Money'),
        ('agent', 'Agent'),
        ('cash', 'Cash'),
    ]
    
    # Vehicle Information
    vehicle_id = models.CharField(max_length=50, unique=True)
    vehicle_type = models.CharField(max_length=20, choices=VEHICLE_TYPES)
    registration_date = models.DateField()
    region = models.CharField(max_length=20, choices=REGIONS)
    
    # Licensing Information
    last_license_renewal = models.DateField(null=True, blank=True)
    is_currently_licensed = models.BooleanField(default=False)
    license_expiry_date = models.DateField(null=True, blank=True)
    
    # Payment Information
    preferred_payment_mode = models.CharField(max_length=20, choices=PAYMENT_MODES)
    total_renewals = models.IntegerField(default=0)
    late_renewals_count = models.IntegerField(default=0)
    average_renewal_delay = models.FloatField(default=0.0)
    
    # Agent Information
    last_agent_sync = models.DateTimeField(null=True, blank=True)
    agent_sync_delay = models.FloatField(default=0.0)  # in hours
    
    # ML Predictions
    risk_score = models.FloatField(null=True, blank=True)
    predicted_unlicensed = models.BooleanField(null=True, blank=True)
    last_prediction_date = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'vehicle_records'
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.vehicle_id} - {self.vehicle_type}"
    
    @property
    def days_since_last_renewal(self):
        if self.last_license_renewal:
            from django.utils import timezone
            return (timezone.now().date() - self.last_license_renewal).days
        return None

class PredictionLog(models.Model):
    vehicle = models.ForeignKey(VehicleRecord, on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateTimeField(auto_now_add=True)
    risk_score = models.FloatField()
    predicted_unlicensed = models.BooleanField()
    model_version = models.CharField(max_length=20)
    features_used = models.JSONField()
    
    class Meta:
        db_table = 'prediction_logs'
        ordering = ['-prediction_date']

class DataUpload(models.Model):
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to='uploads/')
    upload_date = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    records_count = models.IntegerField(default=0)
    processing_notes = models.TextField(blank=True)
    
    class Meta:
        db_table = 'data_uploads'
        ordering = ['-upload_date']