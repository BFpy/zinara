import pandas as pd
from django.conf import settings
from .models import MLModel, ModelTrainingJob
from .ml_pipeline import VehicleLicensePredictor
import os
import logging
from django.utils import timezone
from apps.core.models import VehicleRecord, PredictionLog
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)

def train_new_model(model_type='xgboost', model_name=None, user=None):
    """Train a new ML model"""
    ml_model = None
    training_job = None
    try:
        # Create model name if not provided and ensure uniqueness per version
        if model_name is None:
            base_name = f"{model_type}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            base_name = model_name

        version = '1.0'
        name_candidate = base_name
        suffix = 1
        while MLModel.objects.filter(name=name_candidate, version=version).exists():
            name_candidate = f"{base_name}_{suffix}"
            suffix += 1

        model_name = name_candidate
        creator = user or User.objects.filter(is_superuser=True).first()
        if creator is None:
            raise ValueError("Unable to determine model creator user")

        # Create ML model record
        ml_model = MLModel.objects.create(
            name=model_name,
            model_type=model_type,
            version=version,
            file_path='',
            status='training',
            created_by=creator
        )

        if ml_model.created_by is None:
            raise ValueError("No superuser available to assign as model creator")

        # Create training job
        training_job = ModelTrainingJob.objects.create(
            model=ml_model,
            status='running',
            started_at=timezone.now(),
            training_config={'model_type': model_type}
        )

        # Initialize predictor
        predictor = VehicleLicensePredictor()

        # Load data (use processed enriched dataset)
        data_path = os.path.join(settings.BASE_DIR, 'data', 'processed', 'zinara_vehicle_licensing_enriched.csv')
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
        feature_importance_dict = {str(k): float(v) for k, v in feature_importance}

        # Update model record
        ml_model.file_path = saved_path
        ml_model.status = 'active'
        ml_model.accuracy = float(metrics.get('accuracy')) if metrics.get('accuracy') is not None else None
        ml_model.precision = float(metrics.get('precision')) if metrics.get('precision') is not None else None
        ml_model.recall = float(metrics.get('recall')) if metrics.get('recall') is not None else None
        ml_model.f1_score = float(metrics.get('f1_score')) if metrics.get('f1_score') is not None else None
        ml_model.roc_auc = float(metrics.get('roc_auc')) if metrics.get('roc_auc') is not None else None
        ml_model.threshold = float(metrics.get('threshold', 0.5))
        ml_model.training_data_size = len(df)
        ml_model.feature_importance = feature_importance_dict
        ml_model.confusion_matrix = metrics.get('confusion_matrix')
        ml_model.save()

        # Update training job
        training_job.status = 'completed'
        training_job.completed_at = timezone.now()
        training_job.save()

        logger.info(f"Model {model_name} trained successfully")
        return ml_model, metrics

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

        # Update training job status if it was created
        if training_job is not None:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = timezone.now()
            training_job.save()

        # Update model status if it was created
        if ml_model is not None:
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

        if not active_model.file_path:
            raise FileNotFoundError("Active model file path is not set.")

        model_dir, model_filename = os.path.split(active_model.file_path)
        if not model_filename:
            raise FileNotFoundError("Active model filename is missing.")

        model_name, _ = os.path.splitext(model_filename)
        model_path = model_dir or os.path.join(settings.BASE_DIR, 'data', 'models')

        # Initialize predictor and load model
        predictor = VehicleLicensePredictor()
        predictor.load_model(model_path, model_name)

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
                'vehicle_type': vehicle.vehicle_type,
                'region': vehicle.region,
                'payment_mode': vehicle.preferred_payment_mode,
                'income_level': 'medium',  # Default value for missing income_level
                # Additional base features with defaults
                'late_renewals_count': vehicle.late_renewals_count,
                'agent_hours_used': getattr(vehicle, 'agent_hours_used', 0.0),  # Default
                'user_feedback_score': getattr(vehicle, 'user_feedback_score', 0.5),  # Default neutral
                'predictive_score': getattr(vehicle, 'predictive_score', 0.5),  # Default
                'previous_violations': getattr(vehicle, 'previous_violations', 0),  # Default
                'fine_amount': getattr(vehicle, 'fine_amount', 0.0),  # Default
                'renewal_month': vehicle.last_license_renewal.month if vehicle.last_license_renewal else 6,  # Default June
                'renewal_quarter': vehicle.last_license_renewal.quarter if vehicle.last_license_renewal else 2,  # Default Q2
                'agent_service_used': 'no',  # Default
                'online_platform_used': 'no',  # Default
                'compliance_history': 'good',  # Default
                'geographic_location': vehicle.region,  # Use region as proxy
                # Additional features to match notebook
                'Days Since Last Renewal': vehicle.days_since_last_renewal or 365,
                'Number of Late Renewals in Last 3 Years': vehicle.late_renewals_count,
                'Average Renewal Lag Days': vehicle.average_renewal_delay,
                'Month': vehicle.last_license_renewal.month if vehicle.last_license_renewal else 6,
                'Quarter': vehicle.last_license_renewal.quarter if vehicle.last_license_renewal else 2,
                'Fine Amount': getattr(vehicle, 'fine_amount', 0.0),
                'Previous Violations': getattr(vehicle, 'previous_violations', 0),
                'Predictive Score': getattr(vehicle, 'predictive_score', 0.5),
                'total_vehicles_owned': getattr(vehicle, 'total_vehicles_owned', 1),
                'Number of Vehicles Owned': getattr(vehicle, 'total_vehicles_owned', 1),
                'Agent Hours Used': getattr(vehicle, 'agent_hours_used', 0.0),
                'User Feedback Score': getattr(vehicle, 'user_feedback_score', 0.5),
            }
            vehicle_data.append(vehicle_dict)

        # Create DataFrame
        df = pd.DataFrame(vehicle_data)

        # Standardize columns (same as training)
        df = predictor._standardize_columns(df)

        # Engineer features (same as training)
        df = predictor._engineer_features(df)

        # Encode categorical features consistently if model has stored categories
        if predictor.categorical_categories:
            df = predictor._encode_categoricals_consistently(df)

        # Ensure all expected features exist (including all original features if feature selection was used)
        df = predictor._ensure_all_features_exist(df)

        # For prediction, we need ALL original features (predict() will handle feature selection if needed)
        # Use all_feature_columns if available (before selection), otherwise feature_columns
        features_to_use = predictor.all_feature_columns or predictor.feature_columns
        if not features_to_use:
            raise ValueError("Cannot determine features for prediction")

        # Ensure all required features exist in the dataframe (predict() will create missing ones, but we need the base columns)
        # Don't filter here - pass the full dataframe and let predict() handle feature selection
        # Create a subset with all expected features, filling missing ones with 0
        X = df.reindex(columns=features_to_use, fill_value=0)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

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

def register_notebook_model(model_name, model_type='xgboost', user=None, metrics=None):
    """
    Register a model trained in the notebook to the Django database.
    
    Supports both notebook format (xgboost_model_v1.0.joblib) and Django format ({name}.joblib).
    
    Args:
        model_name: Name of the model file (without .joblib extension), e.g., 'xgboost_model_v1.0'
        model_type: Type of model (default: 'xgboost')
        user: User who created the model (default: first superuser)
        metrics: Optional dict of metrics (accuracy, precision, recall, f1_score, roc_auc)
    
    Returns:
        MLModel instance
    """
    from django.conf import settings
    import os
    import json
    
    model_path = os.path.join(settings.BASE_DIR, 'data', 'models')
    
    # Try Django format first
    model_file = os.path.join(model_path, f"{model_name}.joblib")
    
    # If not found, try notebook format (e.g., xgboost_model_v1.0.joblib)
    if not os.path.exists(model_file):
        # Check if it's in notebook format
        notebook_formats = [
            f"{model_name}.joblib",
            f"{model_type}_model_v1.0.joblib",
            f"{model_type}_model_{model_name}.joblib"
        ]
        for fmt in notebook_formats:
            test_file = os.path.join(model_path, fmt)
            if os.path.exists(test_file):
                model_file = test_file
                model_name = os.path.splitext(os.path.basename(test_file))[0]
                break
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found. Tried: {model_path}/{model_name}.joblib")
    
    # Check if model already exists
    existing = MLModel.objects.filter(name=model_name).first()
    if existing:
        logger.info(f"Model {model_name} already registered. Updating...")
        ml_model = existing
    else:
        creator = user or User.objects.filter(is_superuser=True).first()
        if creator is None:
            raise ValueError("Unable to determine model creator user")
        
        ml_model = MLModel.objects.create(
            name=model_name,
            model_type=model_type,
            version='1.0',
            file_path=model_file,
            status='inactive',  # Start as inactive, user can activate manually
            created_by=creator
        )
    
    # Load model to get metrics if not provided
    if metrics is None:
        try:
            # Try to load metadata (Django format)
            meta_file = os.path.join(model_path, f"{model_name}_meta.joblib")
            if os.path.exists(meta_file):
                import joblib
                meta = joblib.load(meta_file)
                if isinstance(meta, dict):
                    metrics = {
                        'threshold': meta.get('threshold', 0.5),
                    }
            else:
                # Try notebook format (JSON)
                json_meta_file = os.path.join(model_path, f"model_metadata_{model_name.replace('_model', '')}.json")
                if not os.path.exists(json_meta_file):
                    json_meta_file = os.path.join(model_path, "model_metadata_v1.0.json")
                if os.path.exists(json_meta_file):
                    with open(json_meta_file, 'r') as f:
                        meta = json.load(f)
                        if 'metrics' in meta:
                            metrics = meta['metrics']
                            if 'threshold' not in metrics and 'threshold' in meta:
                                metrics['threshold'] = meta.get('threshold', 0.5)
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
    
    # Update metrics if provided
    if metrics:
        if 'accuracy' in metrics:
            ml_model.accuracy = float(metrics['accuracy'])
        if 'precision' in metrics:
            ml_model.precision = float(metrics['precision'])
        if 'recall' in metrics:
            ml_model.recall = float(metrics['recall'])
        if 'f1_score' in metrics:
            ml_model.f1_score = float(metrics['f1_score'])
        if 'roc_auc' in metrics:
            ml_model.roc_auc = float(metrics['roc_auc'])
        if 'threshold' in metrics:
            ml_model.threshold = float(metrics['threshold'])
        if 'confusion_matrix' in metrics:
            ml_model.confusion_matrix = metrics['confusion_matrix']
    
    ml_model.save()
    logger.info(f"Model {model_name} registered successfully")
    return ml_model