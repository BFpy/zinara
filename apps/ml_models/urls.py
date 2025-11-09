from django.urls import path
from . import views

app_name = 'ml_models'

urlpatterns = [
    path('', views.model_list, name='model_list'),
    path('train/', views.train_model, name='train_model'),
    path('predictions/', views.make_predictions, name='make_predictions'),
    path('performance/', views.model_performance, name='performance'),
    path('<int:model_id>/', views.model_detail, name='model_detail'),
    path('<int:model_id>/activate/', views.activate_model, name='activate_model'),
]