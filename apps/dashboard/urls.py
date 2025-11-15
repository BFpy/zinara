from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.overview, name='overview'),
    path('prediction-analysis/', views.prediction_analysis, name='prediction_analysis'),
    path('risk-analysis/', views.risk_analysis, name='risk_analysis'),
	path('upload-analysis/', views.upload_analysis, name='upload_analysis'),
    path('model-explanation/', views.model_explanation, name='model_explanation'),
    path('vehicles/', views.vehicle_list, name='vehicle_list'),
    path('vehicles/<str:vehicle_id>/', views.vehicle_detail, name='vehicle_detail'),
    path('analytics/', views.analytics, name='analytics'),
    path('alerts/', views.alerts_view, name='alerts'),
]