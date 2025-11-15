from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.core.paginator import Paginator
from .models import MLModel, ModelTrainingJob
from .utils import train_new_model, make_predictions_for_all_vehicles, get_model_performance_summary
import json

@login_required
def model_list(request):
    """List all ML models"""
    models = MLModel.objects.all().order_by('-created_at')
    paginator = Paginator(models, 10)

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'models': page_obj.object_list,
    }
    return render(request, 'ml_models/model_list.html', context)

@login_required
def model_detail(request, model_id):
    """Detailed view of a specific model"""
    model = get_object_or_404(MLModel, id=model_id)
    training_jobs = ModelTrainingJob.objects.filter(model=model).order_by('-started_at')

    # Prepare confusion matrix and metrics safely
    conf_matrix = None
    if getattr(model, 'confusion_matrix', None):
        # Expecting dict with 'matrix' key as [[TN, FP], [FN, TP]]
        if isinstance(model.confusion_matrix, dict):
            conf_matrix = model.confusion_matrix.get('matrix') or model.confusion_matrix
        elif isinstance(model.confusion_matrix, list):
            conf_matrix = model.confusion_matrix

    # Format feature importance for display
    feature_importance_items = []
    if model.feature_importance:
        if isinstance(model.feature_importance, dict):
            feature_importance_items = sorted(model.feature_importance.items(), key=lambda item: item[1], reverse=True)
        elif isinstance(model.feature_importance, list):
            feature_importance_items = model.feature_importance

    metrics_available = any([
        model.accuracy,
        model.precision,
        model.recall,
        model.f1_score,
        model.roc_auc,
    ])

    context = {
        'model': model,
        'training_jobs': training_jobs,
        'feature_importance_items': feature_importance_items,
        'confusion_matrix': conf_matrix,
        'threshold': getattr(model, 'threshold', None),
        'metrics_available': metrics_available,
    }
    return render(request, 'ml_models/model_detail.html', context)

@login_required
def train_model(request):
    """Train a new model"""
    if request.method == 'POST':
        model_type = request.POST.get('model_type', 'xgboost')
        model_name = request.POST.get('model_name')

        try:
            model, metrics = train_new_model(
                model_type=model_type,
                model_name=model_name,
                user=request.user
            )

            messages.success(request, f'Model "{model.name}" trained successfully!')
            return redirect('ml_models:model_detail', model_id=model.id)

        except Exception as e:
            messages.error(request, f'Error training model: {str(e)}')

    context = {
        'model_types': MLModel.MODEL_TYPES,
    }
    return render(request, 'ml_models/train_model.html', context)

@login_required
def make_predictions(request):
    """Make predictions for all vehicles"""
    if request.method == 'POST':
        try:
            make_predictions_for_all_vehicles()
            messages.success(request, 'Predictions completed successfully!')
        except Exception as e:
            messages.error(request, f'Error making predictions: {str(e)}')

    return redirect('ml_models:model_list')

@login_required
def model_performance(request):
    """Show model performance comparison"""
    performance_data = get_model_performance_summary()

    context = {
        'performance_data': performance_data,
    }
    return render(request, 'ml_models/model_performance.html', context)

@login_required
def activate_model(request, model_id):
    """Activate or deactivate a specific model"""
    model = get_object_or_404(MLModel, id=model_id)

    if request.method == 'POST':
        if model.status == 'active':
            model.status = 'inactive'
            model.save(update_fields=['status'])
            messages.info(request, f'Model "{model.name}" deactivated.')
        else:
            # Deactivate all other models first
            MLModel.objects.exclude(id=model_id).filter(status='active').update(status='inactive')

            model.status = 'active'
            model.save(update_fields=['status'])
            messages.success(request, f'Model "{model.name}" activated successfully!')

    return redirect('ml_models:model_list')

@login_required
def delete_model(request, model_id):
    """Delete a specific model"""
    model = get_object_or_404(MLModel, id=model_id)

    if request.method == 'POST':
        model_name = model.name
        model.delete()
        messages.success(request, f'Model "{model_name}" deleted successfully!')

    return redirect('ml_models:model_list')
