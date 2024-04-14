from django.shortcuts import render #какой именно html шаблон будем показывать при переходе
#from django.http import HttpResponse
from django.http import JsonResponse, HttpResponse

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pycaret.time_series import *

Path_to_file = ''
Header_row = 0

# Create your views here.
def index(request):
    return render(request,'authorization/authorization.html')

def authorization_check(request):
    #Логика проверки
    return render(request,'doc_upload/doc_upload.html')

def handle_uploaded_file(f):
    with open(f"uploads/{f.name}", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
            
def prepare_data (f):
    global Path_to_file
    global Header_row
    file_path = f"uploads/{f.name}"
    Path_to_file = file_path
    header_row = 5
    Header_row = header_row
    df = pd.read_excel(file_path,header = header_row)
    column_names = df.columns.tolist()
    
    return column_names
    #print(df.head)
    #print(column_names)

def upload_file_to_server(request):
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['fileToUpload'])
        data = prepare_data(request.FILES['fileToUpload'])
        
    return render(request,'create_prediction/model_settings.html', {'columns' : data})

def submit_factors(request):
    if request.method == 'POST':
        selected_values = request.POST.dict()  # Получаем данные из POST-запроса в виде словаря
        # Обрабатываем эти данные
        parameter_values = [value for key, value in selected_values.items() if key.startswith('parameter')]
        target_variable_values = [value for key, value in selected_values.items() if key.startswith('target_variable')]

   
        

        #Выбираем только нужные столбцы
        df = pd.read_excel(Path_to_file, header = Header_row)
        df = df [parameter_values + target_variable_values]
        
        cols = list(df)
        df_for_training = df[:251-Header_row]

        # Создаем экземпляр IterativeImputer
        imputer = IterativeImputer()

        # Преобразуем DataFrame в массив numpy
        data_array = np.array(df_for_training.values)

        # Заполняем пропущенные значения
        imputed_data_array = imputer.fit_transform(data_array)

        # Преобразуем массив обратно в DataFrame
        imputed_df = pd.DataFrame(imputed_data_array, columns=cols)


        

        s = setup(imputed_df, target = target_variable_values[0], fh = 28, fold=5)

        best=compare_models()

        plot_model(best)
        

        return HttpResponse(status=204)

    else:
        return JsonResponse({'error': 'Метод не поддерживается'}, status=405)

