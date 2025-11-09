from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('vehicles/', views.VehicleListAPI.as_view(), name='vehicle_list'),
    path('vehicles/<str:vehicle_id>/', views.VehicleDetailAPI.as_view(), name='vehicle_detail'),
    path('statistics/', views.StatisticsAPI.as_view(), name='statistics'),
    path('predictions/', views.PredictionAPI.as_view(), name='predictions'),
]