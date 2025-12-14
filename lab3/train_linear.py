#!/usr/bin/env python3
"""
Скрипт для обучения линейной регрессии для предсказания цен квартир.
Цель: достичь RMSE < 500,000 с помощью линейных моделей.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def get_rmse(y_true, y_pred):
    """Вычисляет RMSE по формуле из задания"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def preprocess_data_linear(df, is_train=True, label_encoders=None, scaler=None, poly_features=None):
    """
    Расширенная предобработка данных для линейной регрессии:
    - Кодирование категориальных признаков
    - Расширенный feature engineering
    - Полиномиальные признаки
    """
    df = df.copy()
    
    # Базовые числовые признаки
    numeric_features = [
        'kitchen_area', 'bath_area', 'other_area', 'extra_area', 
        'extra_area_count', 'year', 'ceil_height', 'floor_max', 
        'floor', 'total_area', 'bath_count', 'rooms_count'
    ]
    
    # Расширенный feature engineering
    df['area_ratio'] = df['kitchen_area'] / (df['total_area'] + 1e-6)
    df['bath_ratio'] = df['bath_area'] / (df['total_area'] + 1e-6)
    df['other_ratio'] = df['other_area'] / (df['total_area'] + 1e-6)
    df['extra_area_ratio'] = df['extra_area'] / (df['total_area'] + 1e-6)
    df['floor_ratio'] = df['floor'] / (df['floor_max'] + 1e-6)
    df['age'] = 2025 - df['year']  # Возраст квартиры
    df['area_per_room'] = df['total_area'] / (df['rooms_count'] + 1)
    df['bath_per_bathroom'] = df['bath_area'] / (df['bath_count'] + 1e-6)
    df['kitchen_per_total'] = df['kitchen_area'] / (df['total_area'] + 1e-6)
    df['total_rooms'] = df['rooms_count'] + df['bath_count']  # Общее количество комнат
    df['area_per_total_room'] = df['total_area'] / (df['total_rooms'] + 1)
    
    # Взаимодействия важных признаков
    df['total_area_sq'] = df['total_area'] ** 2
    df['total_area_cub'] = df['total_area'] ** 3
    df['age_sq'] = df['age'] ** 2
    df['rooms_area_interaction'] = df['rooms_count'] * df['total_area']
    df['floor_area_interaction'] = df['floor'] * df['total_area']
    df['year_area_interaction'] = df['year'] * df['total_area']
    df['ceil_height_area'] = df['ceil_height'] * df['total_area']
    df['rooms_floor_interaction'] = df['rooms_count'] * df['floor']
    df['rooms_year_interaction'] = df['rooms_count'] * df['year']
    df['floor_year_interaction'] = df['floor'] * df['year']
    df['district_rooms_interaction'] = df['rooms_count'] * (df['district_name_encoded'] if 'district_name_encoded' in df.columns else 0)
    df['district_area_interaction'] = df['total_area'] * (df['district_name_encoded'] if 'district_name_encoded' in df.columns else 0)
    
    # Дополнительные важные взаимодействия
    df['kitchen_bath_ratio'] = df['kitchen_area'] / (df['bath_area'] + 1e-6)
    df['total_extra_ratio'] = (df['total_area'] + df['extra_area']) / (df['total_area'] + 1e-6)
    df['floor_ratio_sq'] = df['floor_ratio'] ** 2
    df['area_per_room_sq'] = df['area_per_room'] ** 2
    
    # Логарифмические преобразования для некоторых признаков
    df['log_total_area'] = np.log1p(df['total_area'])
    df['log_kitchen_area'] = np.log1p(df['kitchen_area'])
    df['log_extra_area'] = np.log1p(df['extra_area'] + 1)
    
    # Категориальные признаки
    categorical_cols = ['gas', 'hot_water', 'central_heating', 
                       'extra_area_type_name', 'district_name']
    
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for test data")
        for col in categorical_cols:
            le = label_encoders[col]
            df[col + '_encoded'] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            ).fillna(-1).astype(int)
    
    # Удаляем исходные категориальные колонки
    df = df.drop(columns=categorical_cols)
    
    # Выбираем все числовые признаки (включая новые взаимодействия)
    feature_cols = [
        'kitchen_area', 'bath_area', 'other_area', 'extra_area', 
        'extra_area_count', 'year', 'ceil_height', 'floor_max', 
        'floor', 'total_area', 'bath_count', 'rooms_count',
        'area_ratio', 'bath_ratio', 'other_ratio', 'extra_area_ratio', 
        'floor_ratio', 'age', 'area_per_room', 'bath_per_bathroom',
        'kitchen_per_total', 'total_rooms', 'area_per_total_room',
        'total_area_sq', 'total_area_cub', 'age_sq', 'rooms_area_interaction',
        'floor_area_interaction', 'year_area_interaction', 'ceil_height_area',
        'rooms_floor_interaction', 'rooms_year_interaction', 'floor_year_interaction',
        'kitchen_bath_ratio', 'total_extra_ratio', 'floor_ratio_sq', 'area_per_room_sq',
        'log_total_area', 'log_kitchen_area', 'log_extra_area',
        'gas_encoded', 'hot_water_encoded', 'central_heating_encoded',
        'extra_area_type_name_encoded', 'district_name_encoded'
    ]
    
    # Убираем признаки, которые еще не созданы (для тестовых данных)
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    
    # Масштабирование
    if scaler is None and is_train:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    elif scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = X
    
    # Полиномиальные признаки (опционально, для некоторых моделей)
    if poly_features is not None:
        X_poly = poly_features.transform(X_scaled)
        X_final = pd.DataFrame(
            X_poly,
            columns=[f'poly_{i}' for i in range(X_poly.shape[1])],
            index=X_scaled.index
        )
    else:
        X_final = X_scaled
    
    if is_train:
        y = df['price'] if 'price' in df.columns else None
        return X_final, y, label_encoders, scaler, poly_features
    else:
        return X_final, label_encoders, scaler, poly_features

def train_linear_models():
    """Обучение линейных моделей"""
    
    print("Загрузка данных...")
    data = pd.read_csv('/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/data.csv')
    test_data = pd.read_csv('/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/test.csv')
    
    print(f"Размер обучающего датасета: {data.shape}")
    print(f"Размер тестового датасета: {test_data.shape}")
    
    # Предобработка обучающих данных
    print("\nПредобработка данных...")
    X, y, label_encoders, scaler, _ = preprocess_data_linear(data, is_train=True)
    
    print(f"Количество признаков после предобработки: {X.shape[1]}")
    
    # Разделение на train/test (70/30) как в MATLAB коде
    print("\nРазделение данных на train/test (70/30)...")
    P = 0.70
    m = len(X)
    np.random.seed(42)
    idx = np.random.permutation(m)
    train_size = round(P * m)
    
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    
    # Попробуем разные линейные модели
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
        'Ridge (alpha=10.0)': Ridge(alpha=10.0, random_state=42),
        'Ridge (alpha=100.0)': Ridge(alpha=100.0, random_state=42),
        'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42, max_iter=2000),
        'Lasso (alpha=10.0)': Lasso(alpha=10.0, random_state=42, max_iter=2000),
        'ElasticNet (alpha=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
        'ElasticNet (alpha=10.0)': ElasticNet(alpha=10.0, l1_ratio=0.5, random_state=42, max_iter=2000),
    }
    
    best_model = None
    best_rmse = float('inf')
    best_model_name = None
    
    print("\nОбучение линейных моделей...")
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = get_rmse(y_test, y_pred)
            print(f"  RMSE на валидации: {rmse:,.2f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"  Ошибка: {e}")
    
    print(f"\n{'='*60}")
    print(f"Лучшая модель: {best_model_name}")
    print(f"RMSE на валидации: {best_rmse:,.2f}")
    print(f"{'='*60}")
    
    # Если RMSE все еще больше 500,000, попробуем полиномиальные признаки
    if best_rmse > 500000:
        print(f"\n⚠️  RMSE ({best_rmse:,.2f}) больше целевого значения 500,000")
        print("Пробуем полиномиальные признаки степени 2...")
        
        # Создаем полиномиальные признаки
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        print(f"Количество признаков после полиномиального преобразования: {X_train_poly.shape[1]}")
        
        # Пробуем Ridge с полиномиальными признаками
        poly_models = {
            'Ridge (poly, alpha=1.0)': Ridge(alpha=1.0, random_state=42),
            'Ridge (poly, alpha=10.0)': Ridge(alpha=10.0, random_state=42),
            'Ridge (poly, alpha=100.0)': Ridge(alpha=100.0, random_state=42),
            'Ridge (poly, alpha=1000.0)': Ridge(alpha=1000.0, random_state=42),
        }
        
        for name, model in poly_models.items():
            print(f"\nОбучение {name}...")
            try:
                model.fit(X_train_poly, y_train)
                y_pred_poly = model.predict(X_test_poly)
                rmse_poly = get_rmse(y_test, y_pred_poly)
                print(f"  RMSE на валидации: {rmse_poly:,.2f}")
                
                if rmse_poly < best_rmse:
                    best_rmse = rmse_poly
                    best_model = model
                    best_model_name = name
                    # Сохраняем полиномиальный преобразователь
                    poly_features = poly
            except Exception as e:
                print(f"  Ошибка: {e}")
    
    # Инициализация переменных для отслеживания использованных техник
    use_log_transform = False
    use_poly_features = False
    poly_features_obj = None
    
    # Если все еще не достигли цели, пробуем логарифмическое преобразование
    if best_rmse > 500000:
        print(f"\n⚠️  RMSE ({best_rmse:,.2f}) все еще больше 500,000")
        print("Пробуем логарифмическое преобразование целевой переменной...")
        
        # Логарифмическое преобразование цены
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        
        # Пробуем Ridge с логарифмической целевой переменной и полиномиальными признаками
        print("\nОбучение Ridge с логарифмом и полиномиальными признаками...")
        try:
            # Создаем полиномиальные признаки
            poly_log = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_train_poly_log = poly_log.fit_transform(X_train)
            X_test_poly_log = poly_log.transform(X_test)
            
            print(f"  Количество признаков после полиномиального преобразования: {X_train_poly_log.shape[1]}")
            
            log_poly_models = {
                'Ridge (log+poly, alpha=100.0)': Ridge(alpha=100.0, random_state=42),
                'Ridge (log+poly, alpha=1000.0)': Ridge(alpha=1000.0, random_state=42),
                'Ridge (log+poly, alpha=10000.0)': Ridge(alpha=10000.0, random_state=42),
            }
            
            for name, model in log_poly_models.items():
                print(f"  Обучение {name}...")
                model.fit(X_train_poly_log, y_train_log)
                y_pred_log = model.predict(X_test_poly_log)
                y_pred_original = np.expm1(y_pred_log)
                rmse_log_poly = get_rmse(y_test, y_pred_original)
                print(f"    RMSE на валидации: {rmse_log_poly:,.2f}")
                
                if rmse_log_poly < best_rmse:
                    best_rmse = rmse_log_poly
                    best_model = model
                    best_model_name = name
                    use_log_transform = True
                    use_poly_features = True
                    poly_features_obj = poly_log
        except Exception as e:
            print(f"  Ошибка: {e}")
        
        # Если все еще не достигли цели, пробуем удаление выбросов + логарифм + полиномы
        if best_rmse > 500000:
            print("\nПробуем удаление выбросов + логарифм + полиномиальные признаки...")
            
            # Удаление выбросов по цене (верхние и нижние 0.5%)
            price_lower = y_train.quantile(0.005)
            price_upper = y_train.quantile(0.995)
            mask = (y_train >= price_lower) & (y_train <= price_upper)
            X_train_clean = X_train[mask]
            y_train_clean = y_train[mask]
            y_train_clean_log = np.log1p(y_train_clean)
            
            print(f"  Размер после удаления выбросов: {X_train_clean.shape[0]}")
            
            # Создаем полиномиальные признаки для очищенных данных
            poly_clean = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_train_clean_poly = poly_clean.fit_transform(X_train_clean)
            X_test_poly_clean = poly_clean.transform(X_test)
            
            print(f"  Количество признаков после полиномиального преобразования: {X_train_clean_poly.shape[1]}")
            
            clean_log_poly_models = {
                'Ridge (clean+log+poly, alpha=100.0)': Ridge(alpha=100.0, random_state=42),
                'Ridge (clean+log+poly, alpha=1000.0)': Ridge(alpha=1000.0, random_state=42),
                'Ridge (clean+log+poly, alpha=10000.0)': Ridge(alpha=10000.0, random_state=42),
            }
            
            for name, model in clean_log_poly_models.items():
                print(f"  Обучение {name}...")
                model.fit(X_train_clean_poly, y_train_clean_log)
                y_pred_log_clean = model.predict(X_test_poly_clean)
                y_pred_original_clean = np.expm1(y_pred_log_clean)
                rmse_clean_log_poly = get_rmse(y_test, y_pred_original_clean)
                print(f"    RMSE на валидации: {rmse_clean_log_poly:,.2f}")
                
                if rmse_clean_log_poly < best_rmse:
                    best_rmse = rmse_clean_log_poly
                    best_model = model
                    best_model_name = name
                    use_log_transform = True
                    use_poly_features = True
                    poly_features_obj = poly_clean
        
        # Если все еще не достигли цели, пробуем без полиномиальных признаков
        if best_rmse > 500000:
            print("\nПробуем Ridge с логарифмом без полиномиальных признаков...")
            log_models = {
                'Ridge (log, alpha=1.0)': Ridge(alpha=1.0, random_state=42),
                'Ridge (log, alpha=10.0)': Ridge(alpha=10.0, random_state=42),
                'Ridge (log, alpha=100.0)': Ridge(alpha=100.0, random_state=42),
            }
            
            for name, model in log_models.items():
                print(f"  Обучение {name}...")
                model.fit(X_train, y_train_log)
                y_pred_log = model.predict(X_test)
                y_pred_original = np.expm1(y_pred_log)
                rmse_log = get_rmse(y_test, y_pred_original)
                print(f"    RMSE на валидации: {rmse_log:,.2f}")
                
                if rmse_log < best_rmse:
                    best_rmse = rmse_log
                    best_model = model
                    best_model_name = name
                    use_log_transform = True
                    use_poly_features = False
    
    # Предсказания на тестовом датасете
    print("\nПредобработка тестового датасета...")
    X_test_final, _, _, _ = preprocess_data_linear(
        test_data, 
        is_train=False, 
        label_encoders=label_encoders, 
        scaler=scaler,
        poly_features=None
    )
    
    # Применяем полиномиальные признаки, если использовали
    if use_poly_features or ('poly' in best_model_name.lower()):
        print("Применяем полиномиальные признаки к тестовому датасету...")
        if poly_features_obj is not None:
            X_test_final_poly = poly_features_obj.transform(X_test_final)
        elif 'poly_features' in locals():
            X_test_final_poly = poly_features.transform(X_test_final)
        else:
            # Создаем новый полиномиальный преобразователь
            poly_temp = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_test_final_poly = poly_temp.fit_transform(X_test_final)
        predictions_temp = best_model.predict(X_test_final_poly)
    else:
        print("Делаем предсказания на тестовом датасете...")
        predictions_temp = best_model.predict(X_test_final)
    
    # Преобразуем обратно из логарифма, если использовали логарифмическое преобразование
    if use_log_transform:
        predictions = np.expm1(predictions_temp)
    else:
        predictions = predictions_temp
    
    # Создаем submission файл
    print("\nСоздание submission файла...")
    submission = pd.DataFrame({
        'index': np.arange(len(predictions)),
        'price': predictions.astype(int)
    })
    
    output_path = '/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/submission_linear.csv'
    submission.to_csv(output_path, index=False)
    print(f"Submission файл сохранен: {output_path}")
    print(f"Размер submission: {submission.shape}")
    print(f"\nПервые 10 предсказаний:")
    print(submission.head(10))
    
    # Финальная оценка на полном обучающем датасете
    print("\n" + "="*60)
    print("Финальная оценка на полном обучающем датасете:")
    
    if use_poly_features or ('poly' in best_model_name.lower()):
        if poly_features_obj is not None:
            X_full_poly = poly_features_obj.transform(X)
        elif 'poly_features' in locals():
            X_full_poly = poly_features.transform(X)
        else:
            poly_temp = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            X_full_poly = poly_temp.fit_transform(X)
        y_train_pred_temp = best_model.predict(X_full_poly)
    else:
        y_train_pred_temp = best_model.predict(X)
    
    if use_log_transform:
        y_train_pred = np.expm1(y_train_pred_temp)
    else:
        y_train_pred = y_train_pred_temp
    
    rmse_full = get_rmse(y, y_train_pred)
    print(f"RMSE на полном датасете: {rmse_full:,.2f}")
    print("="*60)
    
    if best_rmse < 500000:
        print(f"\n✅ УСПЕХ! RMSE ({best_rmse:,.2f}) меньше целевого значения 500,000")
    else:
        print(f"\n⚠️  RMSE ({best_rmse:,.2f}) все еще больше 500,000")
    
    return best_model, best_rmse, label_encoders, scaler

if __name__ == '__main__':
    train_linear_models()

