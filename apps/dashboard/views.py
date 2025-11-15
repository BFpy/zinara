from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q, Avg
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
import json
import os
import pandas as pd
import numpy as np
import logging

from apps.core.models import VehicleRecord, PredictionLog
from apps.ml_models.utils import make_predictions_for_all_vehicles
from apps.ml_models.ml_pipeline import VehicleLicensePredictor
from .models import Alert, DashboardPreferences
from django.views.decorators.http import require_http_methods
from django import forms

logger = logging.getLogger(__name__)

DATA_PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed', 'zinara_vehicle_licensing_enriched.csv')

def _get_model_path_and_name(active_model):
    """Extract model path and name from active_model.file_path"""
    from django.conf import settings
    
    model_path = os.path.join(settings.BASE_DIR, 'data', 'models')
    
    # If file_path is not set or empty, use name
    if not active_model.file_path or active_model.file_path.strip() == '':
        model_name = active_model.name or 'unknown'
        return model_path, model_name
    
    # Normalize the file_path
    file_path = active_model.file_path.strip()
    
    # Check if file_path is just an extension or invalid
    if file_path.startswith('.') or file_path == '.joblib' or len(file_path) < 5:
        # Invalid file_path, use name instead
        model_name = active_model.name or 'unknown'
        return model_path, model_name
    
    # Extract model name from file_path
    model_dir, model_filename = os.path.split(file_path)
    
    # If no filename extracted, use name
    if not model_filename or model_filename == '.joblib' or model_filename.startswith('.'):
        model_name = active_model.name or 'unknown'
        # Check if file_path is a valid directory
        if model_dir and os.path.isdir(model_dir):
            model_path = model_dir
        return model_path, model_name
    
    # Extract name without extension
    model_name, ext = os.path.splitext(model_filename)
    
    # Validate model_name is not empty
    if not model_name or model_name.strip() == '':
        model_name = active_model.name or 'unknown'
    
    # Use the directory from file_path, or default to models directory
    if model_dir and os.path.isdir(model_dir):
        model_path = model_dir
    elif model_dir:
        # If model_dir exists but is not a directory, it might be part of the filename
        # Try to reconstruct
        full_path = os.path.join(settings.BASE_DIR, 'data', 'models', model_filename)
        if os.path.exists(full_path):
            model_path = os.path.join(settings.BASE_DIR, 'data', 'models')
            model_name, _ = os.path.splitext(model_filename)
    
    # Final validation: check if the model file actually exists
    # If not, try to find it by name in the models directory
    expected_file = os.path.join(model_path, f"{model_name}.joblib")
    if not os.path.exists(expected_file):
        # Try to find the model by name in the models directory
        models_dir = os.path.join(settings.BASE_DIR, 'data', 'models')
        if os.path.isdir(models_dir):
            # Try using active_model.name directly
            if active_model.name:
                alt_file = os.path.join(models_dir, f"{active_model.name}.joblib")
                if os.path.exists(alt_file):
                    logger.warning(f"Model file not found at {expected_file}, using {alt_file} instead")
                    return models_dir, active_model.name
            # Try to find any .joblib file that matches the name pattern
            try:
                import glob
                pattern = os.path.join(models_dir, f"*{active_model.name}*.joblib")
                matches = glob.glob(pattern)
                if matches:
                    match_file = matches[0]
                    match_name = os.path.splitext(os.path.basename(match_file))[0]
                    logger.warning(f"Model file not found at {expected_file}, using {match_file} instead")
                    return models_dir, match_name
            except Exception:
                pass
    
    return model_path, model_name

def _load_stats_from_csv():
    """Fallback: compute dashboard stats from processed CSV when DB data is unavailable."""
    if not os.path.exists(DATA_PROCESSED_PATH):
        return None

    df = pd.read_csv(DATA_PROCESSED_PATH)
    df['is_licensed'] = df.get('is_licensed', False).astype(bool)

    total = len(df)
    licensed = int(df['is_licensed'].sum())
    unlicensed = int(total - licensed)

    high_risk = 0
    if 'Predictive Score' in df.columns:
        high_risk = int(((df['Predictive Score'] >= 0.7) & (~df['is_licensed'])).sum())

    regional = []
    if 'Region' in df.columns:
        reg_group = df.groupby('Region').agg(
            total=('Vehicle ID', 'count'),
            licensed=('is_licensed', 'sum')
        ).reset_index()
        reg_group['unlicensed'] = reg_group['total'] - reg_group['licensed']
        regional = [
            {'region': r['Region'], 'total': int(r['total']), 'licensed': int(r['licensed']), 'unlicensed': int(r['unlicensed'])}
            for _, r in reg_group.iterrows()
        ]

    vehicle_type_stats = []
    if 'Vehicle Type' in df.columns:
        type_group = df.groupby('Vehicle Type').agg(
            total=('Vehicle ID', 'count'),
            licensed=('is_licensed', 'sum')
        ).reset_index()
        vehicle_type_stats = [
            {'vehicle_type': r['Vehicle Type'], 'total': int(r['total']), 'licensed': int(r['licensed'])}
            for _, r in type_group.iterrows()
        ]

    return {
        'total_vehicles': total,
        'licensed_vehicles': licensed,
        'unlicensed_vehicles': unlicensed,
        'high_risk_vehicles': high_risk,
        'regional_stats': regional,
        'vehicle_type_stats': vehicle_type_stats,
    }

def _build_vehicle_records(df: pd.DataFrame, source: str = "dataset") -> list:
    """Convert a dataframe into a list of vehicle records suitable for dashboard views."""
    if df is None or df.empty:
        return []

    records = []
    df = df.copy().reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for idx, row in df.iterrows():
        data = row.to_dict()
        vehicle_id = data.get('vehicle_id') or data.get('Vehicle ID') or f"{source.upper()}-{idx+1:05d}"
        vehicle_type_value = str(data.get('vehicle_type') or data.get('Vehicle Type') or 'unknown').strip()
        region_value = str(data.get('region') or data.get('Region') or 'unknown').strip()
        is_licensed = bool(data.get('is_currently_licensed', data.get('is_licensed', False)))
        risk_score = data.get('risk_score', data.get('predictive_score', 0.0))

        if pd.isna(risk_score):
            risk_score = 0.0
        risk_score = float(risk_score)
        risk_score = max(0.0, min(1.0, risk_score))

        if vehicle_type_value:
            display_vehicle_type = vehicle_type_value.replace('_', ' ').title()
        else:
            display_vehicle_type = 'Unknown'

        if region_value:
            display_region = region_value.replace('_', ' ').title()
        else:
            display_region = 'Unknown'

        risk_level = 'low'
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'

        record = {
            'vehicle_id': str(vehicle_id),
            'vehicle_type': display_vehicle_type,
            'vehicle_type_value': vehicle_type_value.lower(),
            'region': display_region,
            'region_value': region_value.lower(),
            'is_currently_licensed': is_licensed,
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'source': source,
        }

        for col in numeric_cols:
            value = data.get(col)
            if pd.notna(value):
                record[col] = float(value)

        records.append(record)

    return records

@login_required
def overview(request):
    """Main dashboard overview - dynamic data from DB, uploaded analysis, or training data"""
    # Priority 1: Check for uploaded/analyzed data in session
    session_data = request.session.get('upload_analysis_data')
    
    # Priority 2: Check database for predictions/records
    total_vehicles = VehicleRecord.objects.count()
    
    # Priority 3: Fallback to training data CSV
    
    if session_data:
        # Use uploaded/analyzed data
        context = {
            'total_vehicles': session_data.get('total_vehicles', 0),
            'licensed_vehicles': session_data.get('licensed_vehicles', 0),
            'unlicensed_vehicles': session_data.get('unlicensed_vehicles', 0),
            'high_risk_vehicles': session_data.get('high_risk_vehicles', 0),
            'compliance_rate': session_data.get('compliance_rate', 0),
            'recent_predictions': [],
            'alerts': Alert.objects.filter(acknowledged=False)[:5],
            'regional_stats': json.dumps(session_data.get('regional_stats', [])),
            'vehicle_type_stats': json.dumps(session_data.get('vehicle_type_stats', [])),
            'data_source': 'uploaded_analysis'
        }
    elif total_vehicles > 0:
        # Use database records (from predictions)
        licensed_vehicles = VehicleRecord.objects.filter(is_currently_licensed=True).count()
        unlicensed_vehicles = total_vehicles - licensed_vehicles
        high_risk_vehicles = VehicleRecord.objects.filter(
            risk_score__gte=0.7, predicted_unlicensed=True
        ).count()
        
        # Recent predictions
        recent_predictions = PredictionLog.objects.select_related('vehicle')[:10]
        
        # Alerts
        unacknowledged_alerts = Alert.objects.filter(acknowledged=False)[:5]
        
        # Regional breakdown
        regional_stats = VehicleRecord.objects.values('region').annotate(
            total=Count('id'),
            licensed=Count('id', filter=Q(is_currently_licensed=True)),
            unlicensed=Count('id', filter=Q(is_currently_licensed=False))
        )
        
        # Vehicle type breakdown
        vehicle_type_stats = VehicleRecord.objects.values('vehicle_type').annotate(
            total=Count('id'),
            licensed=Count('id', filter=Q(is_currently_licensed=True))
        )
        
        context = {
            'total_vehicles': total_vehicles,
            'licensed_vehicles': licensed_vehicles,
            'unlicensed_vehicles': unlicensed_vehicles,
            'high_risk_vehicles': high_risk_vehicles,
            'compliance_rate': round((licensed_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0, 2),
            'recent_predictions': recent_predictions,
            'alerts': unacknowledged_alerts,
            'regional_stats': json.dumps(list(regional_stats)),
            'vehicle_type_stats': json.dumps(list(vehicle_type_stats)),
            'data_source': 'database'
        }
    else:
        # Fallback to training data CSV
        stats = _load_stats_from_csv() or {
            'total_vehicles': 0, 'licensed_vehicles': 0, 'unlicensed_vehicles': 0,
            'high_risk_vehicles': 0, 'regional_stats': [], 'vehicle_type_stats': []
        }
        context = {
            **stats,
            'compliance_rate': round((stats['licensed_vehicles'] / stats['total_vehicles'] * 100) if stats['total_vehicles'] > 0 else 0, 2),
            'recent_predictions': [],
            'alerts': [],
            'regional_stats': json.dumps(stats.get('regional_stats', [])),
            'vehicle_type_stats': json.dumps(stats.get('vehicle_type_stats', [])),
            'data_source': 'training_data'
    }
    return render(request, 'dashboard/overview.html', context)

@login_required
def prediction_analysis(request):
    """Comprehensive prediction analysis with reports and charts"""
    from django.conf import settings
    from apps.ml_models.models import MLModel
    
    # Get active model
    active_model = MLModel.objects.filter(status='active').order_by('-created_at').first()
    
    if not active_model:
        context = {
            'error': 'No active model found. Please train and activate a model first.',
            'has_model': False
        }
        return render(request, 'dashboard/prediction_analysis.html', context)
    
    # Load data from processed CSV
    if not os.path.exists(DATA_PROCESSED_PATH):
        context = {
            'error': 'Processed dataset not found. Please ensure the dataset is available.',
            'has_model': True,
            'has_data': False
        }
        return render(request, 'dashboard/prediction_analysis.html', context)
    
    try:
        # Load model first to get expected feature schema
        predictor = VehicleLicensePredictor()
        model_path, model_name = _get_model_path_and_name(active_model)
        try:
            predictor.load_model(model_path, model_name)
        except Exception as e:
            context = {
                'error': f'Error loading model: {str(e)}. Please ensure the model is trained and saved correctly.',
                'has_model': True,
                'has_data': True
            }
            return render(request, 'dashboard/prediction_analysis.html', context)
        
        # Load and analyze data
        df = predictor.load_and_preprocess_data(DATA_PROCESSED_PATH)
        
        # Encode categorical features using model's stored categories
        if predictor.categorical_categories:
            df = predictor._encode_categoricals_consistently(df)
        
        # Ensure ALL expected features exist (including all original features if feature selection was used)
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
        
        # Get target for comparison (if available)
        y = None
        if 'is_licensed' in df.columns:
            target_series = df['is_licensed'].astype(bool)
            y = (~target_series).astype(int)  # Predict 1 for unlicensed
        
        predictions, probabilities = predictor.predict(X)
        
        # Update model last_used timestamp
        active_model.last_used = timezone.now()
        active_model.save()
        
        # Add predictions to dataframe
        df['risk_score'] = probabilities
        df['predicted_unlicensed'] = predictions.astype(bool)
        
        # Risk categorization
        df['risk_category'] = pd.cut(
            probabilities,
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        # Convert to string to avoid categorical issues when filling with 'N/A'
        df['risk_category'] = df['risk_category'].astype(str)
        
        # Generate comprehensive statistics
        total_vehicles = len(df)
        high_risk_count = int((df['risk_category'] == 'High Risk').sum())
        medium_risk_count = int((df['risk_category'] == 'Medium Risk').sum())
        low_risk_count = int((df['risk_category'] == 'Low Risk').sum())
        
        avg_risk_score = float(probabilities.mean())
        
        # Regional analysis
        regional_analysis = []
        if 'region' in df.columns:
            regional_group = df.groupby('region').agg({
                'risk_score': ['mean', 'count'],
                'predicted_unlicensed': 'sum'
            }).reset_index()
            regional_group.columns = ['region', 'avg_risk', 'total', 'high_risk_count']
            regional_group = regional_group.fillna(0)
            regional_analysis = regional_group.to_dict('records')
        
        # Vehicle type analysis
        vehicle_type_analysis = []
        if 'vehicle_type' in df.columns:
            type_group = df.groupby('vehicle_type').agg({
                'risk_score': ['mean', 'count'],
                'predicted_unlicensed': 'sum'
            }).reset_index()
            type_group.columns = ['vehicle_type', 'avg_risk', 'total', 'high_risk_count']
            type_group = type_group.fillna(0)
            vehicle_type_analysis = type_group.to_dict('records')
        
        # Payment mode analysis
        payment_analysis = []
        if 'payment_mode' in df.columns:
            payment_group = df.groupby('payment_mode').agg({
                'risk_score': 'mean',
                'predicted_unlicensed': 'sum',
                'vehicle_id': 'count'
            }).reset_index()
            payment_group.columns = ['payment_mode', 'avg_risk', 'high_risk_count', 'total']
            payment_group = payment_group.fillna(0)
            payment_analysis = payment_group.to_dict('records')
        
        # Top high-risk vehicles
        high_risk_df = df.nlargest(20, 'risk_score')[
            ['vehicle_id', 'vehicle_type', 'region', 'risk_score', 'predicted_unlicensed']
        ].copy()
        # Convert categorical columns to string before filling with 'N/A'
        for col in high_risk_df.select_dtypes(include=['category']).columns:
            high_risk_df[col] = high_risk_df[col].astype(str)
        high_risk_df = high_risk_df.fillna('N/A')
        high_risk_vehicles = high_risk_df.to_dict('records')
        
        # Risk distribution over time (if date available)
        time_analysis = []
        if 'data_collection_date' in df.columns:
            df['data_collection_date'] = pd.to_datetime(df['data_collection_date'], errors='coerce')
            df_with_date = df.dropna(subset=['data_collection_date'])
            if len(df_with_date) > 0:
                df_with_date['month'] = df_with_date['data_collection_date'].dt.to_period('M')
                monthly_risk = df_with_date.groupby('month').agg({
                    'risk_score': 'mean',
                    'predicted_unlicensed': 'sum',
                    'vehicle_id': 'count'
                }).reset_index()
                monthly_risk['month'] = monthly_risk['month'].astype(str)
                monthly_risk = monthly_risk.fillna(0)
                time_analysis = monthly_risk.to_dict('records')
        
        # Key insights
        insights = []
        if high_risk_count > 0:
            insights.append(f"⚠️ {high_risk_count} vehicles ({high_risk_count/total_vehicles*100:.1f}%) are at high risk of non-compliance")
        
        if regional_analysis:
            highest_risk_region = max(regional_analysis, key=lambda x: x.get('avg_risk', 0))
            insights.append(f"📍 {highest_risk_region.get('region', 'Unknown')} region has the highest average risk score ({highest_risk_region.get('avg_risk', 0):.2f})")
        
        if vehicle_type_analysis:
            highest_risk_type = max(vehicle_type_analysis, key=lambda x: x.get('avg_risk', 0))
            insights.append(f"🚗 {highest_risk_type.get('vehicle_type', 'Unknown')} vehicles have the highest risk ({highest_risk_type.get('avg_risk', 0):.2f})")
        
        if payment_analysis:
            lowest_risk_payment = min(payment_analysis, key=lambda x: x.get('avg_risk', 1.0))
            insights.append(f"💳 {lowest_risk_payment.get('payment_mode', 'Unknown')} payment mode shows lowest risk ({lowest_risk_payment.get('avg_risk', 0):.2f})")
        
        context = {
            'has_model': True,
            'has_data': True,
            'active_model': active_model,
            'model_name': active_model.name,
            'model_type': active_model.model_type,
            'model_accuracy': active_model.accuracy,
            'model_precision': active_model.precision,
            'model_recall': active_model.recall,
            'model_f1': active_model.f1_score,
            'model_trained_date': active_model.created_at,
            'model_last_used': active_model.last_used,
            'total_vehicles': total_vehicles,
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'low_risk_count': low_risk_count,
            'high_risk_percent': round((high_risk_count / total_vehicles * 100) if total_vehicles > 0 else 0, 1),
            'medium_risk_percent': round((medium_risk_count / total_vehicles * 100) if total_vehicles > 0 else 0, 1),
            'low_risk_percent': round((low_risk_count / total_vehicles * 100) if total_vehicles > 0 else 0, 1),
            'avg_risk_score': round(avg_risk_score, 3),
            'regional_analysis': json.dumps(regional_analysis),
            'vehicle_type_analysis': json.dumps(vehicle_type_analysis),
            'payment_analysis': json.dumps(payment_analysis),
            'high_risk_vehicles': high_risk_vehicles[:20],
            'time_analysis': json.dumps(time_analysis),
            'insights': insights,
            'risk_distribution': json.dumps({
                'low': low_risk_count,
                'medium': medium_risk_count,
                'high': high_risk_count
            })
        }
        
    except Exception as e:
        context = {
            'error': f'Error during analysis: {str(e)}',
            'has_model': True,
            'has_data': True
        }
    
    return render(request, 'dashboard/prediction_analysis.html', context)

@login_required
def risk_analysis(request):
    """Detailed risk analysis page showing vehicles at risk"""
    from django.conf import settings
    from apps.ml_models.models import MLModel
    
    # Get active model
    active_model = MLModel.objects.filter(status='active').order_by('-created_at').first()
    
    risk_level = request.GET.get('risk_level', 'all')
    region_filter = request.GET.get('region')
    vehicle_type_filter = request.GET.get('vehicle_type')
    
    if not active_model or not os.path.exists(DATA_PROCESSED_PATH):
        context = {
            'error': 'Model or data not available',
            'vehicles': [],
            'risk_level': risk_level
        }
        return render(request, 'dashboard/risk_analysis.html', context)
    
    try:
        # Load model first to get expected feature schema
        predictor = VehicleLicensePredictor()
        model_path, model_name = _get_model_path_and_name(active_model)
        try:
            predictor.load_model(model_path, model_name)
        except Exception as e:
            context = {
                'error': f'Error loading model: {str(e)}. Please ensure the model is trained and saved correctly.',
                'vehicles': [],
                'risk_level': risk_level
            }
            return render(request, 'dashboard/risk_analysis.html', context)
        
        # Load and analyze data
        df = predictor.load_and_preprocess_data(DATA_PROCESSED_PATH)
        
        # Encode categorical features using model's stored categories
        if predictor.categorical_categories:
            df = predictor._encode_categoricals_consistently(df)
        
        # Ensure ALL expected features exist (including all original features if feature selection was used)
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
        
        predictions, probabilities = predictor.predict(X)
        
        # Update model last_used timestamp
        active_model.last_used = timezone.now()
        active_model.save()
        
        df['risk_score'] = probabilities
        df['predicted_unlicensed'] = predictions.astype(bool)
        df['risk_category'] = pd.cut(
            probabilities,
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        # Convert to string to avoid categorical issues when filling with 'N/A'
        df['risk_category'] = df['risk_category'].astype(str)
        
        # Apply filters
        if risk_level == 'high':
            df = df[df['risk_category'] == 'High Risk']
        elif risk_level == 'medium':
            df = df[df['risk_category'] == 'Medium Risk']
        elif risk_level == 'low':
            df = df[df['risk_category'] == 'Low Risk']
        
        if region_filter and 'region' in df.columns:
            df = df[df['region'] == region_filter]
        
        if vehicle_type_filter and 'vehicle_type' in df.columns:
            df = df[df['vehicle_type'] == vehicle_type_filter]
        
        # Sort by risk score descending
        df = df.sort_values('risk_score', ascending=False)
        
        # Convert to list of dicts for template
        vehicles_df = df.head(100)[
            ['vehicle_id', 'vehicle_type', 'region', 'risk_score', 'predicted_unlicensed', 'risk_category']
        ].copy()
        # Convert categorical columns to string before filling with 'N/A'
        for col in vehicles_df.select_dtypes(include=['category']).columns:
            vehicles_df[col] = vehicles_df[col].astype(str)
        vehicles_df = vehicles_df.fillna('N/A')
        vehicles = vehicles_df.to_dict('records')
        
        # Get unique values for filters
        regions = sorted([r for r in df['region'].unique().tolist() if pd.notna(r)]) if 'region' in df.columns else []
        vehicle_types = sorted([v for v in df['vehicle_type'].unique().tolist() if pd.notna(v)]) if 'vehicle_type' in df.columns else []
        
        context = {
            'active_model': active_model,
            'model_name': active_model.name,
            'model_type': active_model.model_type,
            'vehicles': vehicles,
            'risk_level': risk_level,
            'regions': regions,
            'vehicle_types': vehicle_types,
            'selected_region': region_filter,
            'selected_vehicle_type': vehicle_type_filter,
            'total_count': len(vehicles)
        }
        
    except Exception as e:
        context = {
            'error': f'Error: {str(e)}',
            'vehicles': [],
            'risk_level': risk_level
        }
    
    return render(request, 'dashboard/risk_analysis.html', context)

@login_required
def model_explanation(request):
    """Explain how the ML model works"""
    from apps.ml_models.models import MLModel
    
    active_model = MLModel.objects.filter(status='active').order_by('-created_at').first()
    
    context = {
        'active_model': active_model,
        'model_metrics': {
            'accuracy': active_model.accuracy if active_model else None,
            'precision': active_model.precision if active_model else None,
            'recall': active_model.recall if active_model else None,
            'f1_score': active_model.f1_score if active_model else None,
        } if active_model else None
    }
    
    return render(request, 'dashboard/model_explanation.html', context)

class UploadDatasetForm(forms.Form):
	file = forms.FileField(help_text="Upload a CSV or Excel file (xlsx/xls)")

@login_required
@require_http_methods(["GET", "POST"])
def upload_analysis(request):
	"""Upload a dataset and run the active model to generate an analysis report"""
	from django.conf import settings
	from apps.ml_models.models import MLModel

	form = UploadDatasetForm()
	context = { 'form': form, 'has_result': False }

	if request.method == 'POST':
		form = UploadDatasetForm(request.POST, request.FILES)
		if form.is_valid():
			upload = form.cleaned_data['file']
			filename = upload.name.lower()

			try:
				# Load file into DataFrame
				if filename.endswith('.csv'):
					df = pd.read_csv(upload)
				elif filename.endswith('.xlsx') or filename.endswith('.xls'):
					df = pd.read_excel(upload)
				else:
					context['error'] = 'Unsupported file type. Please upload a CSV or Excel (xlsx/xls) file.'
					return render(request, 'dashboard/upload_analysis.html', context)

				# Load active model first to get expected feature schema
				active_model = MLModel.objects.filter(status='active').order_by('-created_at').first()
				if not active_model:
					context['error'] = 'No active model found. Please train and activate a model first.'
					return render(request, 'dashboard/upload_analysis.html', context)

				# Prepare and standardize
				predictor = VehicleLicensePredictor()
				model_path, model_name = _get_model_path_and_name(active_model)
				predictor.load_model(model_path, model_name)

				df = predictor._standardize_columns(df)
				df = predictor._clean_data(df)
				df = predictor._engineer_features(df)

				# Encode categorical features using model's stored categories
				if predictor.categorical_categories:
					df = predictor._encode_categoricals_consistently(df)

				# Ensure ALL expected features exist (including all original features if feature selection was used)
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

				# Predict
				predictions, probabilities = predictor.predict(X)

				# Attach results
				df['risk_score'] = probabilities
				df['predicted_unlicensed'] = predictions.astype(bool)
				df['risk_category'] = pd.cut(probabilities, bins=[0, 0.4, 0.7, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])
				# Convert to string to avoid categorical issues when filling with 'N/A'
				df['risk_category'] = df['risk_category'].astype(str)

				total_vehicles = len(df)
				high_risk_count = int((df['risk_category'] == 'High Risk').sum())
				medium_risk_count = int((df['risk_category'] == 'Medium Risk').sum())
				low_risk_count = int((df['risk_category'] == 'Low Risk').sum())
				avg_risk_score = float(probabilities.mean()) if total_vehicles > 0 else 0.0

				# Regional analysis (if region available)
				regional_analysis = []
				if 'region' in df.columns:
					regional_group = df.groupby('region').agg({
						'risk_score': ['mean', 'count'],
						'predicted_unlicensed': 'sum'
					}).reset_index()
					regional_group.columns = ['region', 'avg_risk', 'total', 'high_risk_count']
					regional_group = regional_group.fillna(0)
					regional_analysis = regional_group.to_dict('records')

				# Vehicle type analysis
				vehicle_type_analysis = []
				if 'vehicle_type' in df.columns:
					type_group = df.groupby('vehicle_type').agg({
						'risk_score': ['mean', 'count'],
						'predicted_unlicensed': 'sum'
					}).reset_index()
					type_group.columns = ['vehicle_type', 'avg_risk', 'total', 'high_risk_count']
					type_group = type_group.fillna(0)
					vehicle_type_analysis = type_group.to_dict('records')

				# Payment mode analysis
				payment_analysis = []
				if 'payment_mode' in df.columns:
					payment_group = df.groupby('payment_mode').agg({
						'risk_score': 'mean',
						'predicted_unlicensed': 'sum',
						'vehicle_id': 'count' if 'vehicle_id' in df.columns else 'size'
					}).reset_index()
					payment_group.columns = ['payment_mode', 'avg_risk', 'high_risk_count', 'total']
					payment_group = payment_group.fillna(0)
					payment_analysis = payment_group.to_dict('records')

				# Top high-risk
				cols = [c for c in ['vehicle_id', 'vehicle_type', 'region', 'risk_score', 'predicted_unlicensed'] if c in df.columns]
				high_risk_df = df.sort_values('risk_score', ascending=False)[cols].head(50).copy()
				# Convert categorical columns to string before filling with 'N/A'
				for col in high_risk_df.select_dtypes(include=['category']).columns:
					high_risk_df[col] = high_risk_df[col].astype(str)
				high_risk_vehicles = high_risk_df.fillna('N/A').to_dict('records')

				snapshot_columns = [c for c in ['vehicle_id', 'vehicle_type', 'region', 'is_licensed', 'risk_score', 'predictive_score'] if c in df.columns]
				snapshot_df = df[snapshot_columns] if snapshot_columns else df[['risk_score']]
				session_records = _build_vehicle_records(snapshot_df, source='uploaded_dataset')
				if session_records:
					request.session['vehicle_list_source'] = session_records
					request.session['vehicle_list_source_meta'] = {
						'origin': 'uploaded_dataset',
						'updated': timezone.now().isoformat()
					}
					request.session.modified = True

				# Calculate dashboard stats
				licensed_count = int((~df['predicted_unlicensed']).sum()) if 'predicted_unlicensed' in df.columns else 0
				unlicensed_count = int(df['predicted_unlicensed'].sum()) if 'predicted_unlicensed' in df.columns else 0
				
				# Store in session for dashboard
				dashboard_stats = {
					'total_vehicles': total_vehicles,
					'licensed_vehicles': licensed_count,
					'unlicensed_vehicles': unlicensed_count,
					'high_risk_vehicles': high_risk_count,
					'compliance_rate': round((licensed_count / total_vehicles * 100) if total_vehicles > 0 else 0, 2),
					'regional_stats': [
						{'region': r.get('region', 'Unknown'), 'total': int(r.get('total', 0)), 
						 'licensed': int(r.get('total', 0) - r.get('high_risk_count', 0)), 
						 'unlicensed': int(r.get('high_risk_count', 0))}
						for r in regional_analysis
					],
					'vehicle_type_stats': [
						{'vehicle_type': v.get('vehicle_type', 'Unknown'), 'total': int(v.get('total', 0)), 
						 'licensed': int(v.get('total', 0) - v.get('high_risk_count', 0))}
						for v in vehicle_type_analysis
					]
				}
				request.session['upload_analysis_data'] = dashboard_stats
				request.session.modified = True
				
				context.update({
					'has_result': True,
					'total_vehicles': total_vehicles,
					'high_risk_count': high_risk_count,
					'medium_risk_count': medium_risk_count,
					'low_risk_count': low_risk_count,
					'avg_risk_score': round(avg_risk_score, 3),
					'risk_distribution': json.dumps({'low': low_risk_count, 'medium': medium_risk_count, 'high': high_risk_count}),
					'regional_analysis': json.dumps(regional_analysis),
					'vehicle_type_analysis': json.dumps(vehicle_type_analysis),
					'payment_analysis': json.dumps(payment_analysis),
					'high_risk_vehicles': high_risk_vehicles,
				})

			except Exception as e:
				context['error'] = f"Failed to analyze uploaded file: {str(e)}"

		context['form'] = form

	return render(request, 'dashboard/upload_analysis.html', context)

@login_required
def vehicle_list(request):
    """List all vehicles with filtering and smart fallbacks when the DB is empty."""
    region_filter = (request.GET.get('region') or '').strip()
    vehicle_type_filter = (request.GET.get('vehicle_type') or '').strip()
    license_status = (request.GET.get('license_status') or '').strip()
    risk_level = (request.GET.get('risk_level') or '').strip().lower()
    page_number = request.GET.get('page')

    using_fallback = False
    fallback_source = None

    vehicles_qs = VehicleRecord.objects.all()

    if vehicles_qs.exists():
        if region_filter:
            vehicles_qs = vehicles_qs.filter(region__iexact=region_filter)
        if vehicle_type_filter:
            vehicles_qs = vehicles_qs.filter(vehicle_type__iexact=vehicle_type_filter)
        if license_status == 'licensed':
            vehicles_qs = vehicles_qs.filter(is_currently_licensed=True)
        elif license_status == 'unlicensed':
            vehicles_qs = vehicles_qs.filter(is_currently_licensed=False)

        if risk_level == 'high':
            vehicles_qs = vehicles_qs.filter(risk_score__gte=0.7)
        elif risk_level == 'medium':
            vehicles_qs = vehicles_qs.filter(risk_score__gte=0.4, risk_score__lt=0.7)
        elif risk_level == 'low':
            vehicles_qs = vehicles_qs.filter(risk_score__lt=0.4)

        vehicles_qs = vehicles_qs.order_by('-risk_score', 'vehicle_id')
        from django.core.paginator import Paginator
        paginator = Paginator(vehicles_qs, 50)
        page_obj = paginator.get_page(page_number)
    else:
        using_fallback = True
        session_records = request.session.get('vehicle_list_source')
        meta = request.session.get('vehicle_list_source_meta') or {}
        records = session_records if session_records else []
        fallback_source = meta.get('origin')

        if not records:
            if os.path.exists(DATA_PROCESSED_PATH):
                try:
                    df = pd.read_csv(DATA_PROCESSED_PATH)
                    predictor = VehicleLicensePredictor()
                    df = predictor._standardize_columns(df)
                    
                    # Ensure we have the necessary columns
                    required_cols = ['vehicle_id', 'vehicle_type', 'region', 'is_licensed']
                    available_cols = [c for c in required_cols if c in df.columns]
                    
                    # Add predictive_score if available
                    if 'predictive_score' in df.columns:
                        available_cols.append('predictive_score')
                    
                    if available_cols:
                        snapshot_df = df[available_cols].copy()
                        # Calculate risk_score from predictive_score if available
                        if 'predictive_score' in snapshot_df.columns:
                            snapshot_df['risk_score'] = snapshot_df['predictive_score'].fillna(0.0)
                        else:
                            snapshot_df['risk_score'] = 0.0
                        
                        records = _build_vehicle_records(snapshot_df, source='processed_dataset')
                        request.session['vehicle_list_source'] = records
                        request.session['vehicle_list_source_meta'] = {
                            'origin': 'processed_dataset',
                            'updated': timezone.now().isoformat()
                        }
                        request.session.modified = True
                        fallback_source = 'processed_dataset'
                    else:
                        records = []
                except Exception as e:
                    logger.error(f"Error loading vehicle data: {e}")
                    records = []
            else:
                records = []

        region_filter_norm = region_filter.lower()
        vehicle_type_norm = vehicle_type_filter.lower()

        def _matches(record):
            if region_filter_norm and record.get('region_value', record.get('region', '').lower()) != region_filter_norm:
                return False
            if vehicle_type_norm and record.get('vehicle_type_value', record.get('vehicle_type', '').lower()) != vehicle_type_norm:
                return False
            if license_status == 'licensed' and not record.get('is_currently_licensed', False):
                return False
            if license_status == 'unlicensed' and record.get('is_currently_licensed', False):
                return False
            record_risk_level = record.get('risk_level')
            if risk_level and record_risk_level != risk_level:
                return False
            return True

        filtered_records = [rec for rec in records if _matches(rec)]
        from django.core.paginator import Paginator
        paginator = Paginator(filtered_records, 50)
        page_obj = paginator.get_page(page_number)
        fallback_source = fallback_source or 'processed_dataset'
    
    # Format fallback_source for display (replace underscores with spaces)
    display_source = fallback_source.replace('_', ' ').title() if fallback_source else 'processed dataset'
    
    context = {
        'vehicles': page_obj,
        'page_obj': page_obj,
        'using_fallback': using_fallback,
        'fallback_source': display_source,
        'filters': {
            'region': region_filter,
            'vehicle_type': vehicle_type_filter,
            'license_status': license_status,
            'risk_level': risk_level,
        }
    }
    return render(request, 'dashboard/vehicle_list.html', context)

@login_required
def vehicle_detail(request, vehicle_id):
    """Detailed view of a specific vehicle"""
    # First try to get from database
    try:
        vehicle = VehicleRecord.objects.get(vehicle_id=vehicle_id)
        prediction_history = PredictionLog.objects.filter(vehicle=vehicle)[:10]
        
        context = {
            'vehicle': vehicle,
            'prediction_history': prediction_history,
            'from_database': True
        }
        return render(request, 'dashboard/vehicle_detail.html', context)
    except VehicleRecord.DoesNotExist:
        # Vehicle not in database, check fallback sources
        vehicle_data = None
        
        # Check session data first
        session_records = request.session.get('vehicle_list_source')
        if session_records:
            for record in session_records:
                if str(record.get('vehicle_id', '')) == str(vehicle_id):
                    vehicle_data = record
                    break
        
        # If not in session, check CSV file
        if not vehicle_data and os.path.exists(DATA_PROCESSED_PATH):
            try:
                df = pd.read_csv(DATA_PROCESSED_PATH)
                predictor = VehicleLicensePredictor()
                df = predictor._standardize_columns(df)
                
                # Find the vehicle in the CSV
                vehicle_row = None
                if 'vehicle_id' in df.columns:
                    vehicle_row = df[df['vehicle_id'] == vehicle_id]
                elif 'Vehicle ID' in df.columns:
                    vehicle_row = df[df['Vehicle ID'] == vehicle_id]
                
                if not vehicle_row.empty:
                    row = vehicle_row.iloc[0]
                    vehicle_data = {
                        'vehicle_id': str(row.get('vehicle_id') or row.get('Vehicle ID', vehicle_id)),
                        'vehicle_type': str(row.get('vehicle_type') or row.get('Vehicle Type', 'Unknown')),
                        'region': str(row.get('region') or row.get('Region', 'Unknown')),
                        'is_currently_licensed': bool(row.get('is_licensed', row.get('License Status', False))),
                        'risk_score': float(row.get('risk_score', row.get('predictive_score', 0.0))),
                        'predicted_unlicensed': bool(row.get('predicted_unlicensed', False)),
                        'source': 'csv_file'
                    }
            except Exception as e:
                logger.error(f"Error loading vehicle from CSV: {e}")
        
        if vehicle_data:
            # Create a mock vehicle object for template rendering
            class MockVehicle:
                def __init__(self, data):
                    self.vehicle_id = data.get('vehicle_id', vehicle_id)
                    self.vehicle_type = data.get('vehicle_type', 'Unknown')
                    self.region = data.get('region', 'Unknown')
                    self.is_currently_licensed = data.get('is_currently_licensed', False)
                    self.risk_score = data.get('risk_score', 0.0)
                    self.predicted_unlicensed = data.get('predicted_unlicensed', False)
                    self.source = data.get('source', 'unknown')
                    # Add other fields with defaults
                    self.registration_date = None
                    self.last_license_renewal = None
                    self.license_expiry_date = None
                    self.preferred_payment_mode = None
                    self.total_renewals = 0
                    self.late_renewals_count = 0
                    self.average_renewal_delay = 0.0
                    self.last_agent_sync = None
                    self.agent_sync_delay = 0.0
                    self.last_prediction_date = None
                    self.created_at = None
                    self.updated_at = None
                
                @property
                def days_since_last_renewal(self):
                    return None
            
            mock_vehicle = MockVehicle(vehicle_data)
            
            context = {
                'vehicle': mock_vehicle,
                'prediction_history': [],
                'from_database': False
            }
            return render(request, 'dashboard/vehicle_detail.html', context)
        else:
            # Vehicle not found in any source
            from django.http import Http404
            raise Http404(f"Vehicle with ID '{vehicle_id}' not found")

@login_required
def analytics(request):
    """Analytics and reporting page"""
    # Time-based analysis
    thirty_days_ago = timezone.now() - timedelta(days=30)
    
    # Trend data for charts
    daily_predictions = PredictionLog.objects.filter(
        prediction_date__gte=thirty_days_ago
    ).extra(
        select={'day': 'date(prediction_date)'}
    ).values('day').annotate(
        total=Count('id'),
        high_risk=Count('id', filter=Q(predicted_unlicensed=True))
    ).order_by('day')
    
    # Payment mode effectiveness
    payment_mode_stats = VehicleRecord.objects.values('preferred_payment_mode').annotate(
        total=Count('id'),
        avg_delay=Avg('average_renewal_delay'),
        compliance_rate=Avg('is_currently_licensed')
    )
    
    context = {
        'daily_predictions': json.dumps(list(daily_predictions)),
        'payment_mode_stats': json.dumps(list(payment_mode_stats)),
    }
    return render(request, 'dashboard/analytics.html', context)

@login_required
def alerts_view(request):
    """View all alerts"""
    alerts = Alert.objects.all()[:50]
    
    if request.method == 'POST':
        alert_id = request.POST.get('alert_id')
        if alert_id:
            alert = get_object_or_404(Alert, id=alert_id)
            alert.acknowledged = True
            alert.acknowledged_by = request.user
            alert.acknowledged_at = timezone.now()
            alert.save()
            return JsonResponse({'status': 'success'})
    
    context = {
        'alerts': alerts,
    }
    return render(request, 'dashboard/alerts.html', context)
