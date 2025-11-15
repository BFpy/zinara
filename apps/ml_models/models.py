from django.db import models
from django.contrib.auth.models import User
import json

class MLModel(models.Model):
    MODEL_TYPES = [
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree'),
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    version = models.CharField(max_length=20)
    file_path = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    
    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True, help_text="Optimal threshold for predictions")
    
    # Training information
    training_data_size = models.IntegerField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    hyperparameters = models.JSONField(null=True, blank=True)
    confusion_matrix = models.JSONField(null=True, blank=True, help_text="Confusion matrix: [[TN, FP], [FN, TP]]")
    
    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'ml_models'
        ordering = ['-created_at']
        unique_together = ['name', 'version']
    
    def __str__(self):
        return f"{self.name} v{self.version}"

class ModelTrainingJob(models.Model):
    JOB_STATUS = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='training_jobs')
    status = models.CharField(max_length=20, choices=JOB_STATUS, default='pending')
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    training_log = models.TextField(blank=True)
    
    # Training configuration
    training_config = models.JSONField()
    
    class Meta:
        db_table = 'model_training_jobs'
        ordering = ['-started_at']

class FeatureImportance(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=100)
    importance_score = models.FloatField()
    rank = models.IntegerField()
    
    class Meta:
        db_table = 'feature_importance'
        unique_together = ['model', 'feature_name']
        ordering = ['rank']
