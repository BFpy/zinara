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
    training_jobs = ModelTrainingJob.objects.filter(model=model)

    context = {
        'model': model,
        'training_jobs': training_jobs,
        'feature_importance': model.feature_importance or {},
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
    """Activate a specific model"""
    if request.method == 'POST':
        # Deactivate all models
        MLModel.objects.filter(status='active').update(status='inactive')

        # Activate selected model
        model = get_object_or_404(MLModel, id=model_id)
        model.status = 'active'
        model.save()

        messages.success(request, f'Model "{model.name}" activated successfully!')

    return redirect('ml_models:model_list')
