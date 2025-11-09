from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q, Avg
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
import json

from apps.core.models import VehicleRecord, PredictionLog
from .models import Alert, DashboardPreferences

@login_required
def overview(request):
    """Main dashboard overview"""
    # Basic statistics
    total_vehicles = VehicleRecord.objects.count()
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
        'regional_stats': list(regional_stats),
        'vehicle_type_stats': list(vehicle_type_stats),
    }
    return render(request, 'dashboard/overview.html', context)

@login_required
def vehicle_list(request):
    """List all vehicles with filtering"""
    vehicles = VehicleRecord.objects.all()
    
    # Apply filters
    region_filter = request.GET.get('region')
    vehicle_type_filter = request.GET.get('vehicle_type')
    license_status = request.GET.get('license_status')
    risk_level = request.GET.get('risk_level')
    
    if region_filter:
        vehicles = vehicles.filter(region=region_filter)
    if vehicle_type_filter:
        vehicles = vehicles.filter(vehicle_type=vehicle_type_filter)
    if license_status == 'licensed':
        vehicles = vehicles.filter(is_currently_licensed=True)
    elif license_status == 'unlicensed':
        vehicles = vehicles.filter(is_currently_licensed=False)
    if risk_level == 'high':
        vehicles = vehicles.filter(risk_score__gte=0.7)
    elif risk_level == 'medium':
        vehicles = vehicles.filter(risk_score__gte=0.4, risk_score__lt=0.7)
    elif risk_level == 'low':
        vehicles = vehicles.filter(risk_score__lt=0.4)
    
    # Pagination could be added here
    vehicles = vehicles[:100]  # Limit for now
    
    context = {
        'vehicles': vehicles,
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
    vehicle = get_object_or_404(VehicleRecord, vehicle_id=vehicle_id)
    prediction_history = PredictionLog.objects.filter(vehicle=vehicle)[:10]
    
    context = {
        'vehicle': vehicle,
        'prediction_history': prediction_history,
    }
    return render(request, 'dashboard/vehicle_detail.html', context)

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
        'daily_predictions': list(daily_predictions),
        'payment_mode_stats': list(payment_mode_stats),
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
