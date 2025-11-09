from django.db import models
from django.contrib.auth.models import User

class DashboardPreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    default_region_filter = models.CharField(max_length=20, blank=True)
    default_vehicle_type_filter = models.CharField(max_length=20, blank=True)
    notifications_enabled = models.BooleanField(default=True)
    email_alerts = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'dashboard_preferences'

class Alert(models.Model):
    ALERT_TYPES = [
        ('high_risk', 'High Risk Vehicle'),
        ('expired_license', 'Expired License'),
        ('agent_delay', 'Agent Sync Delay'),
        ('system', 'System Alert'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    title = models.CharField(max_length=200)
    message = models.TextField()
    vehicle_id = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'alerts'
        ordering = ['-created_at']
