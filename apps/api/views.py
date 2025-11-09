from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from django.core.paginator import Paginator
from django.db.models import Q
import json
from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from rest_framework import status

from apps.core.models import VehicleRecord, PredictionLog
from apps.dashboard.models import Alert
from .models import APIRequestLog
from .authentication import APIKeyAuthentication

class VehicleListAPI(APIView):
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get list of vehicles with optional filtering"""
        start_time = datetime.now()
        
        # Get query parameters
        page = int(request.GET.get('page', 1))
        per_page = min(int(request.GET.get('per_page', 50)), 100)  # Max 100 per page
        
        # Build filters
        filters = Q()
        if request.GET.get('region'):
            filters &= Q(region=request.GET['region'])
        if request.GET.get('vehicle_type'):
            filters &= Q(vehicle_type=request.GET['vehicle_type'])
        if request.GET.get('is_licensed') is not None:
            filters &= Q(is_currently_licensed=request.GET['is_licensed'].lower() == 'true')
        if request.GET.get('risk_level'):
            if request.GET['risk_level'] == 'high':
                filters &= Q(risk_score__gte=0.7)
            elif request.GET['risk_level'] == 'medium':
                filters &= Q(risk_score__gte=0.4, risk_score__lt=0.7)
            elif request.GET['risk_level'] == 'low':
                filters &= Q(risk_score__lt=0.4)
        
        # Get vehicles
        vehicles = VehicleRecord.objects.filter(filters).order_by('-updated_at')
        
        # Paginate
        paginator = Paginator(vehicles, per_page)
        page_obj = paginator.get_page(page)
        
        # Serialize data
        vehicle_data = []
        for vehicle in page_obj.object_list:
            vehicle_data.append({
                'vehicle_id': vehicle.vehicle_id,
                'vehicle_type': vehicle.vehicle_type,
                'region': vehicle.region,
                'registration_date': vehicle.registration_date.isoformat() if vehicle.registration_date else None,
                'is_currently_licensed': vehicle.is_currently_licensed,
                'last_license_renewal': vehicle.last_license_renewal.isoformat() if vehicle.last_license_renewal else None,
                'license_expiry_date': vehicle.license_expiry_date.isoformat() if vehicle.license_expiry_date else None,
                'risk_score': vehicle.risk_score,
                'predicted_unlicensed': vehicle.predicted_unlicensed,
                'total_renewals': vehicle.total_renewals,
                'late_renewals_count': vehicle.late_renewals_count,
                'preferred_payment_mode': vehicle.preferred_payment_mode,
                'days_since_last_renewal': vehicle.days_since_last_renewal,
            })
        
        # Log API request
        self._log_request(request, start_time, 200)
        
        return Response({
            'success': True,
            'data': vehicle_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
            }
        })

class VehicleDetailAPI(APIView):
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, vehicle_id):
        """Get detailed information for a specific vehicle"""
        start_time = datetime.now()
        
        try:
            vehicle = VehicleRecord.objects.get(vehicle_id=vehicle_id)
            
            # Get prediction history
            predictions = PredictionLog.objects.filter(vehicle=vehicle).order_by('-prediction_date')[:10]
            prediction_history = []
            for pred in predictions:
                prediction_history.append({
                    'prediction_date': pred.prediction_date.isoformat(),
                    'risk_score': pred.risk_score,
                    'predicted_unlicensed': pred.predicted_unlicensed,
                    'model_version': pred.model_version,
                })
            
            vehicle_data = {
                'vehicle_id': vehicle.vehicle_id,
                'vehicle_type': vehicle.vehicle_type,
                'region': vehicle.region,
                'registration_date': vehicle.registration_date.isoformat() if vehicle.registration_date else None,
                'is_currently_licensed': vehicle.is_currently_licensed,
                'last_license_renewal': vehicle.last_license_renewal.isoformat() if vehicle.last_license_renewal else None,
                'license_expiry_date': vehicle.license_expiry_date.isoformat() if vehicle.license_expiry_date else None,
                'risk_score': vehicle.risk_score,
                'predicted_unlicensed': vehicle.predicted_unlicensed,
                'total_renewals': vehicle.total_renewals,
                'late_renewals_count': vehicle.late_renewals_count,
                'average_renewal_delay': vehicle.average_renewal_delay,
                'preferred_payment_mode': vehicle.preferred_payment_mode,
                'agent_sync_delay': vehicle.agent_sync_delay,
                'days_since_last_renewal': vehicle.days_since_last_renewal,
                'prediction_history': prediction_history,
            }
            
            self._log_request(request, start_time, 200)
            return Response({'success': True, 'data': vehicle_data})
            
        except VehicleRecord.DoesNotExist:
            self._log_request(request, start_time, 404)
            return Response({
                'success': False,
                'error': 'Vehicle not found'
            }, status=status.HTTP_404_NOT_FOUND)

class StatisticsAPI(APIView):
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get system statistics"""
        start_time = datetime.now()
        
        # Basic statistics
        total_vehicles = VehicleRecord.objects.count()
        licensed_vehicles = VehicleRecord.objects.filter(is_currently_licensed=True).count()
        unlicensed_vehicles = total_vehicles - licensed_vehicles
        high_risk_vehicles = VehicleRecord.objects.filter(
            risk_score__gte=0.7, predicted_unlicensed=True
        ).count()
        
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
        
        # Recent alerts
        recent_alerts = Alert.objects.filter(
            created_at__gte=datetime.now() - timedelta(days=7)
        ).values('alert_type').annotate(count=Count('id'))
        
        stats_data = {
            'total_vehicles': total_vehicles,
            'licensed_vehicles': licensed_vehicles,
            'unlicensed_vehicles': unlicensed_vehicles,
            'high_risk_vehicles': high_risk_vehicles,
            'compliance_rate': round((licensed_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0, 2),
            'regional_breakdown': list(regional_stats),
            'vehicle_type_breakdown': list(vehicle_type_stats),
            'recent_alerts': list(recent_alerts),
        }
        
        self._log_request(request, start_time, 200)
        return Response({'success': True, 'data': stats_data})

class PredictionAPI(APIView):
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Make predictions for provided vehicle data"""
        start_time = datetime.now()
        
        try:
            # Get vehicle data from request
            vehicle_data = request.data.get('vehicle_data', [])
            if not vehicle_data:
                return Response({
                    'success': False,
                    'error': 'No vehicle data provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Import prediction utility
            from apps.ml_models.utils import make_predictions_for_all_vehicles
            
            # For now, trigger predictions for all vehicles
            # In a production system, you'd want to predict only for the provided data
            make_predictions_for_all_vehicles()
            
            self._log_request(request, start_time, 200)
            return Response({
                'success': True,
                'message': 'Predictions completed successfully'
            })
            
        except Exception as e:
            self._log_request(request, start_time, 500)
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _log_request(self, request, start_time, status_code):
        """Log API request"""
        try:
            response_time = (datetime.now() - start_time).total_seconds()
            
            APIRequestLog.objects.create(
                user=request.user if hasattr(request, 'user') else None,
                endpoint=request.path,
                method=request.method,
                status_code=status_code,
                response_time=response_time,
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                request_data=dict(request.GET) if request.method == 'GET' else (request.data if hasattr(request, 'data') else None),
            )
        except Exception:
            pass  # Don't fail the API call if logging fails
    
    def _get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
