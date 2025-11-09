from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.overview, name='overview'),
    path('vehicles/', views.vehicle_list, name='vehicle_list'),
    path('vehicles/<str:vehicle_id>/', views.vehicle_detail, name='vehicle_detail'),
    path('analytics/', views.analytics, name='analytics'),
    path('alerts/', views.alerts_view, name='alerts'),
]