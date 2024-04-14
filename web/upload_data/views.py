from django.shortcuts import render

# Create your views here.

def upload_data_home(request):
    return render(request, 'create_prediction/model_settings.html')
