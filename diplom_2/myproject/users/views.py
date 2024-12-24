from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden, JsonResponse
from django.conf import settings
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegistrationForm, Predict, ReportsForm
from .models import Transactions, Services, Reports
from django.core.paginator import Paginator
from django.contrib import messages
from django.utils.dateparse import parse_date

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'users', 'best_model_3.keras')
model = load_model(MODEL_PATH)

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('users:login')
    else:
        form = RegistrationForm()
    return render(request, 'users/register.html', {'form': form})

def login(request):
    return render(request, 'users/login.html')

def index(request):
    return render(request, 'users/index.html')

@login_required
def test(request):
    # 1. Обработка POST-запроса (добавление отчёта), как у вас было
    if request.method == 'POST' and 'add_report' in request.POST:
        form = ReportsForm(request.POST)
        if form.is_valid():
            # Создаём объект Reports, но пока не сохраняем в БД
            report = form.save(commit=False)
            # Привязываем к текущему пользователю
            report.user = request.user
            report.save()
            messages.success(request, "Отчёт успешно добавлен!")
            return redirect('users:test')  # или на другую страницу
        else:
            messages.error(request, "Произошла ошибка при сохранении отчёта.")
    else:
        form = ReportsForm()

    # -------------------------------------------------------
    # 2. Фильтрация отчётов
    # -------------------------------------------------------
    user_reports = Reports.objects.filter(user=request.user)

    # Параметры из GET-запроса
    report_name = request.GET.get('report_name', '').strip()
    report_date = request.GET.get('report_date', '').strip()

    # Фильтр по имени отчёта
    if report_name:
        user_reports = user_reports.filter(name__icontains=report_name)

    # Фильтр по точной дате отчёта (формат YYYY-MM-DD)
    if report_date:
        user_reports = user_reports.filter(date=report_date)

    # -------------------------------------------------------
    # 3. Фильтрация транзакций
    # -------------------------------------------------------
    # Здесь мы получаем все транзакции (поскольку в модели нет поля user).
    # Если появится поле user (ForeignKey), то можно будет фильтровать только "свои" транзакции.
    user_transactions = Transactions.objects.all()

    # Параметры для фильтрации транзакций (например, по bin и по дню/месяцу)
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()

    if transaction_bin:
        # Фильтрация по bin (частичное совпадение)
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
    
    # Если заданы Day и Month, фильтруем точным совпадением строк
    if transaction_day:
        user_transactions = user_transactions.filter(Day=transaction_day)
    if transaction_month:
        user_transactions = user_transactions.filter(Month=transaction_month)

    context = {
        'form': form,
        'user_reports': user_reports,
        'user_transactions': user_transactions,
    }

    return render(request, 'users/test.html', context)


def about(request):
    return render(request, 'users/about.html')

def get_time_of_day(seconds):
    if 5*3600 <= seconds < 12*3600:
        return 'morning'
    elif 12*3600 <= seconds < 17*3600:
        return 'evening'
    else:
        return 'night'

def process_creationdate(df):
    df['creationdate'] = pd.to_datetime(df['creationdate'])
    df['day'] = df['creationdate'].dt.day
    df['month'] = df['creationdate'].dt.month
    return df

def prepare_input(data):
    df = pd.DataFrame([data])

    # Drop unnecessary columns if present
    df = df.drop(columns=['txid', 'card_id', 'currencycode'], errors='ignore')

    # Map 'cardverificationcodesupplied' from 'yes'/'no' to 1/0
    df['cardverificationcodesupplied'] = df['cardverificationcodesupplied'].map({'yes': 1, 'no': 0}).fillna(1).astype(int)

    # Create 'is_foreign_transaction'
    df['is_foreign_transaction'] = (df['issuercountrycode'] != df['shoppercountrycode']).astype(int)

    # Process 'creationdate' if present
    if 'creationdate' in df.columns:
        df = process_creationdate(df)

    # Calculate 'bin_length'
    df['bin_length'] = df['bin'].astype(str).apply(len)
    print("!!!!!!!!!!!!!!!!!!!!!")
    print(df['bin_length'].values)
    # Calculate 'amount_to_country_avg'
    country_avg_amount = df.groupby('shoppercountrycode')['amount'].transform('mean')
    df['amount_to_country_avg'] = np.round(df['amount'] / country_avg_amount, 2).fillna(0)

    # Calculate 'amount_to_card_type_avg'
    card_type_avg_amount = df.groupby('txvariantcode')['amount'].transform('mean')
    df['amount_to_card_type_avg'] = np.round(df['amount'] / card_type_avg_amount, 2).fillna(0)

    # Calculate 'cvcres_avg_amount'
    cvcres_avg_amount = df.groupby('cvcresponsecode')['amount'].transform('mean')
    df['cvcres_avg_amount'] = cvcres_avg_amount.fillna(0)

    # Calculate 'cardver_avg_amount'
    cardver_avg_amount = df.groupby('cardverificationcodesupplied')['amount'].transform('mean')
    df['cardver_avg_amount'] = cardver_avg_amount.fillna(0)

    # Calculate 'is_large_transaction'
    df['is_large_transaction'] = (df['amount'] > 5000).astype(int)

    # Calculate 'is_small_transaction'
    df['is_small_transaction'] = (((df['amount'] > 500) & (df['amount'] < 1500)) | 
                                   ((df['amount'] > 3000) & (df['amount'] < 4000))).astype(int)

    # Calculate 'is_holiday_season'
    df['is_holiday_season'] = df['month'].isin([1, 12]).astype(int)

    # Determine 'time_of_day' and create dummies
    df['time_of_day'] = df['time_in_seconds'].apply(get_time_of_day)
    df = pd.get_dummies(df, columns=['time_of_day'], drop_first=True)

    # Calculate 'bin_ver_percentaget'
    bin_ver_percentage = df.groupby('bin')['cardverificationcodesupplied'].transform('mean')
    df['bin_ver_percentaget'] = np.round(bin_ver_percentage, 2).fillna(0)

    # Calculate 'bin_cvs_res_code'
    bin_cvs_res_code = df.groupby('cvcresponsecode')['bin'].transform('mean')
    df['bin_cvs_res_code'] = np.round(bin_cvs_res_code, 2).fillna(0)

    # Calculate 'bin_avg_amount'
    bin_avg_amount = df.groupby('bin')['amount'].transform('mean')
    df['bin_avg_amount'] = np.round(bin_avg_amount, 2).fillna(0)

    # One-hot encode 'shoppercountrycode', 'txvariantcode', 'issuercountrycode'
    shopper_dummies = pd.get_dummies(df['shoppercountrycode'], prefix='shoppercountrycode')
    txvariant_dummies = pd.get_dummies(df['txvariantcode'], prefix='txvariantcode')
    issuercountry_dummies = pd.get_dummies(df['issuercountrycode'], prefix='issuercountrycode')

    # Concatenate dummies with the main dataframe
    df = pd.concat([df, shopper_dummies, txvariant_dummies, issuercountry_dummies], axis=1)

    # Drop original categorical columns
    df = df.drop(columns=['shoppercountrycode', 'txvariantcode', 'issuercountrycode'], errors='ignore')

    # Define all required features (84 features excluding 'is_fraud')
    required_features = [
        'bin', 'amount', 'day', 'month', 'time_in_seconds', 'bin_length',
        'amount_to_country_avg', 'amount_to_card_type_avg', 'bin_avg_amount',
        'bin_ver_percentaget', 'bin_cvs_res_code', 'cvcres_avg_amount',
        'cardver_avg_amount', 'cardverificationcodesupplied',
        'cvcresponsecode', 'is_foreign_transaction', 'is_large_transaction',
        'is_small_transaction', 'is_holiday_season', 'time_of_day_evening',
        'time_of_day_morning', 'time_of_day_night',
        # Issuer country codes
        'issuercountrycode_AR', 'issuercountrycode_AT', 'issuercountrycode_AU',
        'issuercountrycode_BR', 'issuercountrycode_CA', 'issuercountrycode_CL',
        'issuercountrycode_CO', 'issuercountrycode_DE', 'issuercountrycode_EG',
        'issuercountrycode_ES', 'issuercountrycode_ET', 'issuercountrycode_FR',
        'issuercountrycode_GB', 'issuercountrycode_GR', 'issuercountrycode_HR',
        'issuercountrycode_IT', 'issuercountrycode_MA', 'issuercountrycode_MX',
        'issuercountrycode_NI', 'issuercountrycode_NL', 'issuercountrycode_NO',
        'issuercountrycode_NZ', 'issuercountrycode_PL', 'issuercountrycode_PT',
        'issuercountrycode_RO', 'issuercountrycode_SE', 'issuercountrycode_SO',
        'issuercountrycode_UA', 'issuercountrycode_US', 'issuercountrycode_VE',
        # Transaction variant codes
        'txvariantcode_MCDEBIT', 'txvariantcode_VISA',
        'txvariantcode_VISABUSINESS', 'txvariantcode_VISACLASSIC',
        'txvariantcode_VISADEBIT',
        # Shopper country codes
        'shoppercountrycode_AT', 'shoppercountrycode_AU', 'shoppercountrycode_BR',
        'shoppercountrycode_CA', 'shoppercountrycode_CL', 'shoppercountrycode_CO',
        'shoppercountrycode_DE', 'shoppercountrycode_EG', 'shoppercountrycode_ES',
        'shoppercountrycode_ET', 'shoppercountrycode_FR', 'shoppercountrycode_GB',
        'shoppercountrycode_GR', 'shoppercountrycode_HR', 'shoppercountrycode_IT',
        'shoppercountrycode_MA', 'shoppercountrycode_MX', 'shoppercountrycode_NI',
        'shoppercountrycode_NL', 'shoppercountrycode_NZ', 'shoppercountrycode_PT',
        'shoppercountrycode_RO', 'shoppercountrycode_SE', 'shoppercountrycode_SO',
        'shoppercountrycode_UA', 'shoppercountrycode_US', 'shoppercountrycode_VE'
    ]

    # Ensure all required features are present
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    df = df.drop(columns=['shoppercountrycode_UA', 'shoppercountrycode_US'], errors='ignore')
    # Reorder columns to match the required feature order
    #df = df[required_features]

    # Convert all features to float32
    df = df.astype(float)

    # Debugging: Print the prepared DataFrame
    print("Prepared Input DataFrame:")
    print(df.head())
    print(f"Input Data Shape: {df.shape}")

    return df.values

def predict_transactions(request):
    if request.method == 'POST':
        form = Predict(request.POST)
        if form.is_valid():
            try:
                # Сначала сохраняем введённые пользователем данные в таблицу Transactions
                # Обратите внимание, что в модели Transactions поля типа CharField, 
                # поэтому нужно привести их к строкам.
                transaction = Transactions.objects.create(
                    bin=str(form.cleaned_data.get('bin', '0')),
                    amount=str(form.cleaned_data.get('amount', '0')),
                    shoppercountrycode=form.cleaned_data.get('shoppercountrycode', ''),
                    cardverificationcodesupplied=form.cleaned_data.get('cardverificationcodesupplied', 'yes'),
                    cvcresponsecode=str(form.cleaned_data.get('cvcresponsecode', '0')),
                    txvariantcode=form.cleaned_data.get('txvariantcode', ''),
                    Day=str(form.cleaned_data.get('Day', '0')),
                    Month=str(form.cleaned_data.get('Month', '0')),
                    time_in_seconds=str(form.cleaned_data.get('time_in_seconds', '0')),
                    issuercountrycode=form.cleaned_data.get('issuercountrycode', '')
                )

                # Для прогноза модели нам нужны числовые типы, поэтому собираем 
                # аналогичный словарь (но уже с преобразованием к int/float для prepare_input).
                data = {
                    'bin': int(form.cleaned_data.get('bin', 0)),
                    'amount': float(form.cleaned_data.get('amount', 0)),
                    'shoppercountrycode': form.cleaned_data.get('shoppercountrycode', ''),
                    'cardverificationcodesupplied': form.cleaned_data.get('cardverificationcodesupplied', 'yes'),
                    'cvcresponsecode': int(form.cleaned_data.get('cvcresponsecode', 0)),
                    'txvariantcode': form.cleaned_data.get('txvariantcode', ''),
                    'day': int(form.cleaned_data.get('Day', 0)),
                    'month': int(form.cleaned_data.get('Month', 0)),
                    'time_in_seconds': float(form.cleaned_data.get('time_in_seconds', 0)),
                    'issuercountrycode': form.cleaned_data.get('issuercountrycode', '')
                }

                # Debug: распечатать сырые данные
                print("Raw Input Data from Form:")
                print(data)

                input_data = prepare_input(data)

                # Debug: распечатать подготовленные данные
                print("Input Data Passed to Model:")
                print(input_data)
                print(f"Input Data Shape: {input_data.shape}")

                prediction = model.predict(input_data)
                predicted_class = prediction[0][0]

                # Debug: вывод результата предсказания
                print(f"Model Prediction: {predicted_class}")

                return JsonResponse({'prediction': float(predicted_class)})

            except Exception as e:
                print(f"Prediction Error: {e}")
                return JsonResponse({'error': str(e)}, status=400)
        else:
            print("Form is invalid:")
            print(form.errors)
            return JsonResponse({'error': 'Invalid form data'}, status=400)
    else:
        form = Predict()
    return render(request, 'users/predict.html', {'form': form})


def services(request):
    services = Services.objects.all()
    
    # Filter by name
    name = request.GET.get('name', '').strip()
    if name:
        services = services.filter(name__icontains=name)
    
    # Filter by description
    description = request.GET.get('desc', '').strip()
    if description:
        services = services.filter(desc__icontains=description)
    
    # Filter by dates
    start_date = request.GET.get('start_date', '').strip()
    end_date = request.GET.get('end_date', '').strip()
    
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
            services = services.filter(updated_at__date__gte=start_date_obj)
        except ValueError:
            messages.error(request, "Неверный формат даты начала. Используйте формат YYYY-MM-DD.")
    
    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
            services = services.filter(updated_at__date__lte=end_date_obj)
        except ValueError:
            messages.error(request, "Неверный формат даты окончания. Используйте формат YYYY-MM-DD.")
    
    # Pagination
    paginator = Paginator(services, 10)  # 10 services per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'users/serv.html', {'page_obj': page_obj})
