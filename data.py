import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('fraud_detection_model.h5')

# Определение количества признаков, ожидаемых моделью
input_shape = model.input_shape[1]  # Количество ожидаемых признаков
print(input_shape)

# Генерация случайных данных
def generate_random_data(num_samples, num_features, categorical_features=None, num_categories=5):
    categorical_features = categorical_features or []
    data = {}

    for i in range(num_features):
        if i in categorical_features:
            # Генерация случайных категорий
            data[f'feature_{i}'] = np.random.randint(0, num_categories, num_samples)
        else:
            # Генерация случайных чисел в диапазоне [0, 1]
            data[f'feature_{i}'] = np.random.random(num_samples)

    return pd.DataFrame(data)

# Безопасное преобразование категориальных данных
def safe_transform(encoder, column):
    """
    Обрабатывает новые метки, которые не были в обучающем наборе.
    """
    known_labels = set(encoder.classes_)
    new_labels = set(column.unique()) - known_labels

    if new_labels:
        print(f"Warning: New labels encountered: {new_labels}")
        # Добавляем новые метки в LabelEncoder
        encoder.classes_ = np.append(encoder.classes_, list(new_labels))
    return encoder.transform(column)

# Настройки генерации данных
num_samples = 10
num_features = input_shape  # Количество признаков, соответствующих модели
categorical_features = [0, 10]  # Индексы категориальных признаков (пример)

# Генерация данных
random_data = generate_random_data(num_samples, num_features, categorical_features)

# Пример использования LabelEncoder для категориальных признаков
encoders = {}
for col in categorical_features:
    encoder = LabelEncoder()
    encoder.fit(random_data[f'feature_{col}'])
    random_data[f'feature_{col}'] = safe_transform(encoder, random_data[f'feature_{col}'])
    encoders[col] = encoder

# Нормализация данных 
random_data = (random_data - random_data.mean()) / random_data.std()

# Преобразование данных в формат numpy
random_data_np = random_data.to_numpy()

# Предсказание модели
predictions = model.predict(random_data_np)

# Преобразование предсказаний в проценты
predictions_percent = (predictions * 100).round(2)

# Вывод только предсказаний в процентах
print(predictions_percent)

# Сохранение предсказаний в CSV
pd.DataFrame(predictions_percent, columns=['Prediction (%)']).to_csv('predictions.csv', index=False)
