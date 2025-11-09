import pandas as pd
from django.conf import settings
from .models import MLModel, ModelTrainingJob
from .ml_pipeline import VehicleLicensePredictor
import os
import logging
from django.utils import timezone
from apps.core.models import VehicleRecord, PredictionLog

logger = logging.getLogger(__name__)

def train_new_model(model_type='xgboost', model_name=None, user=None):
    """Train a new ML model"""
    try:
        # Create model name if not provided
        if model_name is None:
            model_name = f"{model_type}_{timezone.now().strftime('%Y%m%d_%H%M')}"

        # Create ML model record
        ml_model = MLModel.objects.create(
            name=model_name,
            model_type=model_type,
            version='1.0',
            file_path='',
            status='training',
            created_by=user or User.objects.get(is_superuser=True)
        )

        # Create training job
        training_job = ModelTrainingJob.objects.create(
            model=ml_model,
            status='running',
            started_at=timezone.now(),
            training_config={'model_type': model_type}
        )

        # Initialize predictor
        predictor = VehicleLicensePredictor()

        # Load data
        data_path = os.path.join(settings.BASE_DIR, 'data', 'raw', 'vehicle_licensing_data.csv')
        df = predictor.load_and_preprocess_data(data_path)

        # Prepare features and target
        X, y = predictor.prepare_features_target(df)

        # Train model
        metrics = predictor.train_model(X, y, model_type=model_type)

        # Save model
        model_path = os.path.join(settings.BASE_DIR, 'data', 'models')
        saved_path = predictor.save_model(model_path, model_name)

        # Get feature importance
        feature_importance = predictor.get_feature_importance()

        # Update model record
        ml_model.file_path = saved_path
        ml_model.status = 'active'
        ml_model.accuracy = metrics['accuracy']
        ml_model.precision = metrics['precision']
        ml_model.recall = metrics['recall']
        ml_model.f1_score = metrics['f1_score']
        ml_model.roc_auc = metrics['roc_auc']
        ml_model.training_data_size = len(df)
        ml_model.feature_importance = dict(feature_importance)
        ml_model.save()

        # Update training job
        training_job.status = 'completed'
        training_job.completed_at = timezone.now()
        training_job.save()

        logger.info(f"Model {model_name} trained successfully")
        return ml_model, metrics

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

        # Update training job status
        training_job.status = 'failed'
        training_job.error_message = str(e)
        training_job.completed_at = timezone.now()
        training_job.save()

        # Update model status
        ml_model.status = 'failed'
        ml_model.save()

        raise e

def make_predictions_for_all_vehicles():
    """Make predictions for all vehicles using the active model"""
    try:
        # Get active model
        active_model = MLModel.objects.filter(status='active').order_by('-created_at').first()
        if not active_model:
            logger.error("No active model found")
            return

        # Initialize predictor and load model
        predictor = VehicleLicensePredictor()
        model_path = os.path.join(settings.BASE_DIR, 'data', 'models')
        predictor.load_model(model_path, active_model.name)

        # Get all vehicles
        vehicles = VehicleRecord.objects.all()

        # Prepare vehicle data for prediction
        vehicle_data = []
        for vehicle in vehicles:
            # Convert vehicle data to format expected by model
            vehicle_dict = {
                'days_since_renewal': vehicle.days_since_last_renewal or 365,
                'vehicle_age_years': (timezone.now().date() - vehicle.registration_date).days / 365.25,
                'total_renewals': vehicle.total_renewals,
                'late_renewals_count': vehicle.late_renewals_count,
                'average_renewal_delay': vehicle.average_renewal_delay,
                'agent_sync_delay': vehicle.agent_sync_delay,
                # Add other features as needed
            }
            vehicle_data.append(vehicle_dict)

        # Create DataFrame
        df = pd.DataFrame(vehicle_data)

        # Engineer features (same as training)
        df = predictor._engineer_features(df)

        # Select features for prediction
        X = df[predictor.feature_columns].fillna(0)

        # Make predictions
        predictions, probabilities = predictor.predict(X)

        # Update vehicle records
        for i, vehicle in enumerate(vehicles):
            vehicle.predicted_unlicensed = bool(predictions[i])
            vehicle.risk_score = probabilities[i]
            vehicle.last_prediction_date = timezone.now()
            vehicle.save()

            # Create prediction log
            PredictionLog.objects.create(
                vehicle=vehicle,
                risk_score=probabilities[i],
                predicted_unlicensed=bool(predictions[i]),
                model_version=active_model.version,
                features_used=dict(zip(predictor.feature_columns, X.iloc[i].values))
            )

        # Update model last used
        active_model.last_used = timezone.now()
        active_model.save()

        logger.info(f"Predictions completed for {len(vehicles)} vehicles")

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise e

def get_model_performance_summary():
    """Get performance summary of all models"""
    models = MLModel.objects.all().order_by('-created_at')

    performance_data = []
    for model in models:
        performance_data.append({
            'name': model.name,
            'type': model.model_type,
            'status': model.status,
            'accuracy': model.accuracy,
            'precision': model.precision,
            'recall': model.recall,
            'f1_score': model.f1_score,
            'roc_auc': model.roc_auc,
            'created_at': model.created_at,
            'last_used': model.last_used,
        })

    return performance_data