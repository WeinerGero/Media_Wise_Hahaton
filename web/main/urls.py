
from django.urls import path 
from . import views #для того, чтобы вызвать определенный метод из views при переходе на главную стр

#index без круглых скобок, потому что мы ъотим обратится к нему а не вызвать
urlpatterns = [
    path('', views.index),
    path('upload_file/', views.authorization_check, name='authorization_check'),
    path('create_model/', views.upload_file_to_server, name='model_settings'),
    path('submit_factors', views.submit_factors, name='submit_factors')
]

