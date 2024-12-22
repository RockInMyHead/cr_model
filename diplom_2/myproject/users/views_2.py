from django.shortcuts import render
from django.contrib.auth.views import LoginView, LogoutView
from django.shortcuts import render, redirect
import numpy as np

from users.models import Transactions
from .forms import RegistrationForm
from .forms import Predict
import os
from django.conf import settings
from tensorflow.keras.models import load_model
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden


MODEL_PATH = os.path.join(settings.BASE_DIR, 'users', 'best_model_3.keras')
model = load_model(MODEL_PATH)

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('users:login')  # перенаправление на страницу входа после регистрации
    else:
        form = RegistrationForm()
    return render(request, 'users/register.html', {'form': form})

def login (request):
    return render(request, 'users/login.html')

def  index (request):
    return render(request, 'users/index.html')

def test (request):
   return render(request, 'users/test.html')   

def about (request):
   return render(request, 'users/about.html')  

def predict_transactions (request):
   if request.method == 'POST':
        form = Predict(request.POST)
        if form.is_valid():
            predict_instance = form.save(commit=False)
            bin = form.cleaned_data['bin']
            amount = form.cleaned_data['amount']
            shoppercountrycode = form.cleaned_data['shoppercountrycode']
            cardverificationcodesupplied = form.cleaned_data['cardverificationcodesupplied']
            cvcresponsecode = form.cleaned_data['cvcresponsecode']
            txvariantcode = form.cleaned_data['txvariantcode']
            Day = form.cleaned_data['Day']
            Month = form.cleaned_data['Month']
            time_in_seconds = form.cleaned_data['time_in_seconds']
            issuercountrycode = form.cleaned_data['issuercountrycode']
            features = np.array([[bin, amount,shoppercountrycode,cardverificationcodesupplied,cvcresponsecode,txvariantcode,Day,Month,time_in_seconds,issuercountrycode ]])
            prediction = model.predict(features)
            predict_instance.bin = bin
            predict_instance.save()
            print(prediction)
            return render(request, 'users/transactions.html') 
        else: 
            form = Predict()
   return render(request, 'users/transactions.html')  


@login_required
def admin_self_data(request):
    if not request.user.is_staff:
        return HttpResponseForbidden("Недостаточно прав для доступа.")
    return render(request, 'admin/self_data.html', {'user': request.user})