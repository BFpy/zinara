from django.contrib import admin
from .models import MLModel, ModelTrainingJob, FeatureImportance

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'version', 'status', 'accuracy', 'f1_score', 'created_at']
    list_filter = ['model_type', 'status', 'created_at']
    search_fields = ['name', 'version']
    readonly_fields = ['created_at', 'last_used']

@admin.register(ModelTrainingJob)
class ModelTrainingJobAdmin(admin.ModelAdmin):
    list_display = ['model', 'status', 'started_at', 'completed_at']
    list_filter = ['status', 'started_at']

@admin.register(FeatureImportance)
class FeatureImportanceAdmin(admin.ModelAdmin):
    list_display = ['model', 'feature_name', 'importance_score', 'rank']
    list_filter = ['model']
