from django.urls import path

from .import views
from django.conf.urls.static import static

urlpatterns = [
    path('', views.upload_data_home, name='upload_data_home')
    
] 
