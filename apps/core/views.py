from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from django.contrib import messages
from django.db.models import Count, Q
from .models import VehicleRecord, PredictionLog

def home(request):
    """Home page with basic statistics"""
    if request.user.is_authenticated:
        return redirect('dashboard:overview')
    
    # Public statistics for non-authenticated users
    total_vehicles = VehicleRecord.objects.count()
    licensed_vehicles = VehicleRecord.objects.filter(is_currently_licensed=True).count()
    unlicensed_vehicles = total_vehicles - licensed_vehicles
    
    context = {
        'total_vehicles': total_vehicles,
        'licensed_vehicles': licensed_vehicles,
        'unlicensed_vehicles': unlicensed_vehicles,
        'compliance_rate': round((licensed_vehicles / total_vehicles * 100) if total_vehicles > 0 else 0, 2)
    }
    return render(request, 'core/home.html', context)

class CustomLoginView(auth_views.LoginView):
    template_name = 'auth/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return '/dashboard/'

class CustomLogoutView(auth_views.LogoutView):
    next_page = '/'

def about(request):
    return render(request, 'core/about.html')