from datetime import datetime
from django.shortcuts import render, redirect,  get_object_or_404
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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from django.http import HttpResponse
import openpyxl
from openpyxl.styles import Font

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
    # 1. Обработка POST-запроса для добавления отчёта
    if request.method == 'POST' and 'add_report' in request.POST:
        form = ReportsForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.user = request.user
            report.save()
            messages.success(request, "Отчёт успешно добавлен!")
            return redirect('users:test')
        else:
            messages.error(request, "Произошла ошибка при сохранении отчёта.")
    else:
        form = ReportsForm()

    # 2. Фильтрация отчётов
    user_reports = Reports.objects.filter(user=request.user)
    report_name = request.GET.get('report_name', '').strip()
    report_date = request.GET.get('report_date', '').strip()
    if report_name:
        user_reports = user_reports.filter(name__icontains=report_name)
    if report_date:
        user_reports = user_reports.filter(date=report_date)

    # 3. Фильтрация транзакций
    user_transactions = Transactions.objects.all()
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()
    if transaction_bin:
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
    if transaction_day:
        user_transactions = user_transactions.filter(Day=transaction_day)
    if transaction_month:
        user_transactions = user_transactions.filter(Month=transaction_month)

    # 4. Проверка роли пользователя. Если роль установлена и ее имя равно "one",
    # то устанавливаем флаг show_elements в True
    show_elements = False
    if request.user.role:
        show_elements = (request.user.role.name == "one")

    context = {
        'form': form,
        'user_reports': user_reports,
        'user_transactions': user_transactions,
        'show_elements': show_elements,  # передаем флаг в шаблон
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


def generate():
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    from .models import Transactions

    # ===== 1. Подготовка данных и справочников =====
    # Путь к модели 
    model_path = r"D:\diplom_django\cr_model-main\diplom_2\myproject\users\fraud_detection_model.h5"
    model = load_model(model_path)
    expected_features = model.input_shape[1]
    print(f"Модель ожидает {expected_features} признаков.")

    # Сколько записей генерируем
    num_samples = 5

    # Справочники для более осмысленных данных
    month_dict = {
        1:  'Январь',  2:  'Февраль',   3:  'Март',
        4:  'Апрель',  5:  'Май',       6:  'Июнь',
        7:  'Июль',    8:  'Август',    9:  'Сентябрь',
        10: 'Октябрь', 11: 'Ноябрь',    12: 'Декабрь'
    }

    country_dict = {
        840: 'США',
        643: 'Россия',
        250: 'Франция',
        356: 'Индия',
        36:  'Венгрия',
        276: 'Германия'
    }

    tx_variant_dict = {
        101: 'Visa',
        102: 'Mastercard',
        103: 'Maestro',
        104: 'Mir'
    }

    cvc_dict = {
        0: 'CVC_0',
        1: 'CVC_1',
        2: 'CVC_2',
        3: 'CVC_3',
        4: 'CVC_4',
        5: 'CVC_5'
    }

    # Генерация числовых данных (перед преобразованием в текст)
    numeric_data = pd.DataFrame({
        # BIN от 400000 до 500000 (6-значный номер)
        'bin': np.random.randint(400000, 500000, size=num_samples),

        # Выбор случайного кода страны из country_dict (ключи)
        'shoppercountrycode': np.random.choice(list(country_dict.keys()), size=num_samples),

        # Тип карты из tx_variant_dict
        'txvariantcode': np.random.choice(list(tx_variant_dict.keys()), size=num_samples),

        # Код страны-эмитента (из тех же ключей)
        'issuercountrycode': np.random.choice(list(country_dict.keys()), size=num_samples),

        # Сумма транзакции от 100 до 10,000
        'amount': np.random.randint(100, 10001, size=num_samples),

        # День от 1 до 28
        'Day': np.random.randint(1, 29, size=num_samples),

        # Месяц от 1 до 12
        'Month': np.random.randint(1, 13, size=num_samples),

        # Время в секундах [0..86399]
        'time_in_seconds': np.random.randint(0, 86400, size=num_samples),

        # 0 или 1 (наличие CVC)
        'cardverificationcodesupplied': np.random.randint(0, 2, size=num_samples),

        # ответ системы CVC
        'cvcresponsecode': np.random.choice(list(cvc_dict.keys()), size=num_samples)
    })

    # Признак bin_length: длина BIN (чаще 6)
    numeric_data['bin_length'] = numeric_data['bin'].astype(str).apply(len)

    # ===== 2. Преобразование числовых данных в осмысленные текстовые =====
    # (Месяц -> «Январь», «Февраль» и т.д.)
    numeric_data['Month'] = numeric_data['Month'].apply(lambda x: month_dict[x])
    # (Коды стран -> «США», «Россия»...)
    numeric_data['shoppercountrycode'] = numeric_data['shoppercountrycode'].apply(lambda x: country_dict[x])
    numeric_data['issuercountrycode'] = numeric_data['issuercountrycode'].apply(lambda x: country_dict[x])
    # (txvariantcode -> «Visa», «Mastercard», ...)
    numeric_data['txvariantcode'] = numeric_data['txvariantcode'].apply(lambda x: tx_variant_dict[x])
    # (cvcresponsecode -> «CVC_0», «CVC_1», ...)
    numeric_data['cvcresponsecode'] = numeric_data['cvcresponsecode'].apply(lambda x: cvc_dict[x])

    import copy
    original_data = copy.deepcopy(numeric_data)  # уже в себе содержит целочисленные поля

    df_for_model = pd.DataFrame()
    feature_cols = [
        'bin',
        'amount',
        'shoppercountrycode',
        'cardverificationcodesupplied',
        'cvcresponsecode',
        'txvariantcode',
        'Day',
        'Month',
        'time_in_seconds',
        'issuercountrycode',
        'bin_length'
    ]
    df_for_model[feature_cols] = np.random.randint(0, 1000, size=(num_samples, len(feature_cols)))

    # Добавим «шум», умножая выбранные столбцы:
    features_to_multiply = [
        'shoppercountrycode',
        'cardverificationcodesupplied',
        'cvcresponsecode',
        'txvariantcode',
        'Day',
        'Month',
        'time_in_seconds',
        'issuercountrycode'
    ]
    for col in features_to_multiply:
        multiplier = np.random.uniform(1.0, 10.0)
        df_for_model[col] = df_for_model[col] * multiplier

    # Предсказание
    model_input_np = df_for_model.to_numpy()
    predictions = model.predict(model_input_np)

    # Аргмакс, если больше 1 выхода
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predicted_values = np.argmax(predictions, axis=1)
    else:
        predicted_values = predictions.flatten()
    total = len(predicted_values)
    count_to_replace = np.random.randint(1, total + 1)
    indices_to_replace = np.random.choice(total, size=count_to_replace, replace=False)
    for idx in indices_to_replace:
        predicted_values[idx] = np.random.randint(1, 100)

    # Маппинг числового предсказания на строки
    def map_prediction(val):
        if val == 0:
            return "Не мошенническая"
        elif val == 1:
            return "Мошенническая"
        else:
            return "Не известно"

    numeric_data['Prediction'] = predicted_values
    numeric_data['Prediction_result'] = numeric_data['Prediction'].apply(map_prediction)

    print("Сгенерированные «текстовые» данные с предсказанием:")
    print(numeric_data)

    # ===== 5. Сохраняем в CSV, БД =====
    # При желании сохраните в CSV
    numeric_data.to_csv('predictions.csv', index=False, encoding='utf-8')
    for idx, row in numeric_data.iterrows():
        Transactions.objects.create(
            bin=str(row['bin']),
            amount=str(row['amount']),
            shoppercountrycode=str(row['shoppercountrycode']),
            cardverificationcodesupplied=str(row['cardverificationcodesupplied']),
            cvcresponsecode=str(row['cvcresponsecode']),
            txvariantcode=str(row['txvariantcode']),
            Day=str(row['Day']),
            Month=str(row['Month']),  # Например, «Январь», «Февраль»...
            time_in_seconds=str(row['time_in_seconds']),
            issuercountrycode=str(row['issuercountrycode']),
            result=row['Prediction_result']
        )
    numeric_data['amount'] = numeric_data['amount'].astype(float)
    report_by_month = numeric_data.groupby('Month').agg(
        total_transactions=('bin', 'count'),
        total_amount=('amount', 'sum')
    )
    report_by_month.to_csv('report_by_month.csv', encoding='utf-8')
    print("\n=== Отчёт по месяцам ===")
    print(report_by_month)

    # Отчёт по странам (shoppercountrycode)
    report_by_country = numeric_data.groupby('shoppercountrycode').agg(
        total_transactions=('bin', 'count'),
        total_amount=('amount', 'sum')
    )
    report_by_country.to_csv('report_by_country.csv', encoding='utf-8')
    print("\n=== Отчёт по странам (покупатель) ===")
    print(report_by_country)

    # Отчёт по результату предсказания
    report_by_result = numeric_data.groupby('Prediction_result').agg(
        count=('bin', 'count'),
        average_amount=('amount', 'mean')
    )
    report_by_result.to_csv('report_by_result.csv', encoding='utf-8')
    print("\n=== Отчёт по результату транзакции ===")
    print(report_by_result)

    return predicted_values, numeric_data







def generated(request):
    predict, data  = generate()
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
    user_transactions = Transactions.objects.all()

    # Параметры для фильтрации транзакций (например, по bin и по дню/месяцу)
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()

    if transaction_bin:
        # Фильтрация по bin (частичное совпадение)
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
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


def tex(request):
        # 1. Обработка POST-запроса для добавления отчёта
    if request.method == 'POST' and 'add_report' in request.POST:
        form = ReportsForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.user = request.user
            report.save()
            messages.success(request, "Отчёт успешно добавлен!")
            return redirect('users:test')
        else:
            messages.error(request, "Произошла ошибка при сохранении отчёта.")
    else:
        form = ReportsForm()

    # 2. Фильтрация отчётов
    user_reports = Reports.objects.filter(user=request.user)
    report_name = request.GET.get('report_name', '').strip()
    report_date = request.GET.get('report_date', '').strip()
    if report_name:
        user_reports = user_reports.filter(name__icontains=report_name)
    if report_date:
        user_reports = user_reports.filter(date=report_date)

    # 3. Фильтрация транзакций
    user_transactions = Transactions.objects.all()
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()
    if transaction_bin:
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
    if transaction_day:
        user_transactions = user_transactions.filter(Day=transaction_day)
    if transaction_month:
        user_transactions = user_transactions.filter(Month=transaction_month)

    # 4. Проверка роли пользователя. Если роль установлена и ее имя равно "one",
    # то устанавливаем флаг show_elements в True
    show_elements = False
    if request.user.role:
        show_elements = (request.user.role.name == "one")

    context = {
        'form': form,
        'user_reports': user_reports,
        'user_transactions': user_transactions,
        'show_elements': show_elements,  # передаем флаг в шаблон
    }
    return render(request, 'users/tex.html', context)


def test_one(request):
    # 1. Обработка POST-запроса для добавления отчёта
    if request.method == 'GET' and 'add_report' in request.POST:
        form = ReportsForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.user = request.user
            report.save()
            messages.success(request, "Отчёт успешно добавлен!")
            return redirect('users:test')
        else:
            messages.error(request, "Произошла ошибка при сохранении отчёта.")
    else:
        form = ReportsForm()

    # 2. Фильтрация отчётов
    user_reports = Reports.objects.filter(user=request.user)
    report_name = request.GET.get('report_name', '').strip()
    report_date = request.GET.get('report_date', '').strip()
    if report_name:
        user_reports = user_reports.filter(name__icontains=report_name)
    if report_date:
        user_reports = user_reports.filter(date=report_date)

    # 3. Фильтрация транзакций
    user_transactions = Transactions.objects.all()
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()
    if transaction_bin:
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
    if transaction_day:
        user_transactions = user_transactions.filter(Day=transaction_day)
    if transaction_month:
        user_transactions = user_transactions.filter(Month=transaction_month)

    # 4. Проверка роли пользователя. Если роль установлена и ее имя равно "one",
    # то устанавливаем флаг show_elements в True
    show_elements = False
    if request.user.role:
        show_elements = (request.user.role.name == "one")

    context = {
        'form': form,
        'user_reports': user_reports,
        'user_transactions': user_transactions,
        'show_elements': show_elements,  # передаем флаг в шаблон
    }

    return render(request, 'users/tex_one.html', context)


def tex_one(request):
    # 1. Обработка POST-запроса для добавления отчёта
    if request.method == 'POST' and 'add_report' in request.POST:
        form = ReportsForm(request.POST)
        if form.is_valid():
            report = form.save(commit=False)
            report.user = request.user
            report.save()
            messages.success(request, "Отчёт успешно добавлен!")
            return redirect('users:test')
        else:
            messages.error(request, "Произошла ошибка при сохранении отчёта.")
    else:
        form = ReportsForm()

    # 2. Фильтрация отчётов
    user_reports = Reports.objects.filter(user=request.user)
    report_name = request.GET.get('report_name', '').strip()
    report_date = request.GET.get('report_date', '').strip()
    if report_name:
        user_reports = user_reports.filter(name__icontains=report_name)
    if report_date:
        user_reports = user_reports.filter(date=report_date)

    # 3. Фильтрация транзакций
    #  Здесь result = "Не известно"
    user_transactions = Transactions.objects.filter(result="Не известно")
    transaction_bin = request.GET.get('transaction_bin', '').strip()
    transaction_day = request.GET.get('transaction_day', '').strip()
    transaction_month = request.GET.get('transaction_month', '').strip()
    if transaction_bin:
        user_transactions = user_transactions.filter(bin__icontains=transaction_bin)
    if transaction_day:
        user_transactions = user_transactions.filter(Day=transaction_day)
    if transaction_month:
        user_transactions = user_transactions.filter(Month=transaction_month)

    # 4. Проверка роли пользователя
    show_elements = False
    if request.user.role:
        show_elements = (request.user.role.name == "one")

    context = {
        'form': form,
        'user_reports': user_reports,
        'user_transactions': user_transactions,
        'show_elements': show_elements,
    }
    return render(request, 'users/tex_two.html', context)


def update_transaction(request, transaction_id):
    """Обработчик изменения поля result для конкретной транзакции."""
    if request.method == "POST":
        transaction = get_object_or_404(Transactions, pk=transaction_id)
        new_result = request.POST.get('result')
        # Обновим поле result и сохраним
        transaction.result = new_result
        transaction.save()
        # Можно добавить сообщение об успешном обновлении:
        messages.success(request, f"Транзакция {transaction_id} успешно обновлена!")
    return redirect('users:tex_one')  # возвращаем на страницу со списком


@login_required
def reports_view(request):
    """
    Страница «Отчёты», где можно:
    1) Отфильтровать транзакции.
    2) Посмотреть простую аналитику.
    3) Сформировать Excel-файл.
    """
    # 1. Фильтрация транзакций
    transactions = Transactions.objects.all()

    # Получаем GET-параметры для фильтра
    bin_value = request.GET.get('bin_value', '').strip()
    month_value = request.GET.get('month_value', '').strip()

    if bin_value:
        transactions = transactions.filter(bin__icontains=bin_value)
    if month_value:
        transactions = transactions.filter(Month=month_value)

    # 2. Пример простой аналитики
    total_count = transactions.count()  # количество выбранных транзакций
    sum_amount = 0
    for t in transactions:
        try:
            sum_amount += float(t.amount)
        except ValueError:
            pass  # если какие-то значения не распарсить

    context = {
        'transactions': transactions,
        'total_count': total_count,
        'sum_amount': sum_amount,
    }
    return render(request, 'users/reports.html', context)


@login_required
def download_excel_report(request):
    """
    Генерация и скачивание Excel-файла с разной аналитикой,
    зависящей от выбранного вида отчёта (by_country, by_month, и т.д.).
    """
    # 1. Фильтрация по GET-параметрам
    transactions = Transactions.objects.all()
    bin_value = request.GET.get('bin_value', '').strip()
    month_value = request.GET.get('month_value', '').strip()

    if bin_value:
        transactions = transactions.filter(bin__icontains=bin_value)
    if month_value:
        transactions = transactions.filter(Month=month_value)

    # 2. Собираем данные транзакций в DataFrame для удобства
    data = []
    for t in transactions:
        data.append({
            'bin': t.bin,
            'amount': t.amount,
            'shoppercountrycode': t.shoppercountrycode,
            'Day': t.Day,
            'Month': t.Month,
            'result': t.result,
        })
    df = pd.DataFrame(data)

    # Преобразуем amount в float (если в базе это хранится как str)
    # чтобы можно было складывать
    def to_float_safe(val):
        """Преобразуем val к float, если возможно, иначе 0."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    df['amount'] = df['amount'].apply(to_float_safe)

    # 3. Узнаём тип отчёта
    report_type = request.GET.get('report_type', 'by_country')

    # Создаём новую рабочую книгу Excel
    wb = openpyxl.Workbook()

    # В зависимости от вида отчёта — разная логика
    if report_type == 'by_country':
        # ======== ОТЧЁТ ПО СТРАНАМ ========
        fraud_df = df[df['result'] == 'Мошенническая']

        # Считаем кол-во мошеннических операций по каждой стране
        # и сортируем по убыванию
        report_df = fraud_df.groupby('shoppercountrycode').agg(
            fraud_count=('bin', 'count'),
            total_fraud_amount=('amount', 'sum')
        ).reset_index().sort_values('fraud_count', ascending=False)

        # Создаём лист "Отчёт по странам"
        ws = wb.active
        ws.title = "Страны"

        # Запишем заголовки
        headers = ["Страна", "Кол-во мошеннических транзакций", "Сумма мошеннических транзакций"]
        for col_num, h in enumerate(headers, 1):
            ws.cell(row=1, column=col_num, value=h)

        # Заполняем строки
        for row_idx, row_data in enumerate(report_df.values, start=2):
            country_name = row_data[0]
            fraud_count = row_data[1]
            fraud_amount = row_data[2]
            ws.cell(row=row_idx, column=1, value=country_name)
            ws.cell(row=row_idx, column=2, value=fraud_count)
            ws.cell(row=row_idx, column=3, value=fraud_amount)

        # Название файла для скачивания
        filename = "report_by_country.xlsx"

    elif report_type == 'by_month':
        # ======== ОТЧЁТ ПО МЕСЯЦАМ ========
        fraud_df = df[df['result'] == 'Не мошенническая']

        # Считаем кол-во мошенничеств по месяцам
        report_df = fraud_df.groupby('Month').agg(
            fraud_count=('bin', 'count'),
            total_fraud_amount=('amount', 'sum')
        ).reset_index().sort_values('Month')

        ws = wb.active
        ws.title = "По месяцам"

        headers = ["Месяц", "Кол-во мошеннических транзакций", "Сумма мошеннических транзакций"]
        for col_num, h in enumerate(headers, 1):
            ws.cell(row=1, column=col_num, value=h)

        for row_idx, row_data in enumerate(report_df.values, start=2):
            month_name = row_data[0]
            fraud_count = row_data[1]
            fraud_amount = row_data[2]
            ws.cell(row=row_idx, column=1, value=month_name)
            ws.cell(row=row_idx, column=2, value=fraud_count)
            ws.cell(row=row_idx, column=3, value=fraud_amount)

        filename = "report_by_month.xlsx"

    else:

        ws = wb.active
        ws.title = "Общий отчёт"

        headers = df.columns.tolist()  # ['bin', 'amount', 'shoppercountrycode', ...]
        for col_num, h in enumerate(headers, 1):
            ws.cell(row=1, column=col_num, value=h)

        for row_idx, row_data in enumerate(df.values, start=2):
            for col_num, cell_val in enumerate(row_data, start=1):
                ws.cell(row=row_idx, column=col_num, value=cell_val)

        filename = "report_general.xlsx"

    # 4. Формируем ответ, отдаём файл
    response = HttpResponse(
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    wb.save(response)
    return response