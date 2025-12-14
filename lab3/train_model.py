#!/usr/bin/env python3
"""
Скрипт для предобработки данных и обучения модели предсказания цен квартир.
Цель: достичь RMSE < 500,000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def get_rmse(y_true, y_pred):
    """Вычисляет RMSE по формуле из задания"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def preprocess_data(df, is_train=True, label_encoders=None, scaler=None):
    """
    Предобработка данных:
    - Кодирование категориальных признаков
    - Feature engineering
    - Масштабирование
    """
    df = df.copy()
    
    # Создаем копии для feature engineering
    df['area_ratio'] = df['kitchen_area'] / (df['total_area'] + 1e-6)
    df['bath_ratio'] = df['bath_area'] / (df['total_area'] + 1e-6)
    df['extra_area_ratio'] = df['extra_area'] / (df['total_area'] + 1e-6)
    df['floor_ratio'] = df['floor'] / (df['floor_max'] + 1e-6)
    df['age'] = 2025 - df['year']  # Возраст квартиры
    df['area_per_room'] = df['total_area'] / (df['rooms_count'] + 1)
    df['bath_per_bathroom'] = df['bath_area'] / (df['bath_count'] + 1e-6)
    
    # Категориальные признаки
    categorical_cols = ['gas', 'hot_water', 'central_heating', 
                       'extra_area_type_name', 'district_name']
    
    if is_train:
        # Создаем новые энкодеры для обучающих данных
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        # Используем существующие энкодеры для тестовых данных
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for test data")
        for col in categorical_cols:
            le = label_encoders[col]
            # Обработка новых категорий (если есть)
            df[col + '_encoded'] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            ).fillna(-1).astype(int)
    
    # Удаляем исходные категориальные колонки
    df = df.drop(columns=categorical_cols)
    
    # Выбираем числовые признаки для модели
    feature_cols = [
        'kitchen_area', 'bath_area', 'other_area', 'extra_area', 
        'extra_area_count', 'year', 'ceil_height', 'floor_max', 
        'floor', 'total_area', 'bath_count', 'rooms_count',
        'area_ratio', 'bath_ratio', 'extra_area_ratio', 'floor_ratio',
        'age', 'area_per_room', 'bath_per_bathroom',
        'gas_encoded', 'hot_water_encoded', 'central_heating_encoded',
        'extra_area_type_name_encoded', 'district_name_encoded'
    ]
    
    X = df[feature_cols].copy()
    
    # Масштабирование (опционально, для некоторых моделей не нужно)
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
    
    if is_train:
        y = df['price'] if 'price' in df.columns else None
        return X_scaled, y, label_encoders, scaler
    else:
        return X_scaled, label_encoders, scaler

def train_and_evaluate():
    """Основная функция для обучения и оценки модели"""
    
    print("Загрузка данных...")
    data = pd.read_csv('/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/data.csv')
    test_data = pd.read_csv('/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/test.csv')
    
    print(f"Размер обучающего датасета: {data.shape}")
    print(f"Размер тестового датасета: {test_data.shape}")
    
    # Предобработка обучающих данных
    print("\nПредобработка данных...")
    X, y, label_encoders, scaler = preprocess_data(data, is_train=True)
    
    # Разделение на train/test (70/30) как в MATLAB коде
    print("\nРазделение данных на train/test (70/30)...")
    P = 0.70
    m = len(X)
    np.random.seed(42)  # Для воспроизводимости
    idx = np.random.permutation(m)
    train_size = round(P * m)
    
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    
    # Попробуем несколько моделей
    models = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_rmse = float('inf')
    best_model_name = None
    
    print("\nОбучение моделей...")
    for name, model in models.items():
        print(f"\nОбучение {name}...")
        model.fit(X_train, y_train)
        
        # Предсказания на валидационной выборке
        y_pred = model.predict(X_test)
        rmse = get_rmse(y_test, y_pred)
        
        print(f"  RMSE на валидации: {rmse:,.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
    
    print(f"\n{'='*60}")
    print(f"Лучшая модель: {best_model_name}")
    print(f"RMSE на валидации: {best_rmse:,.2f}")
    print(f"{'='*60}")
    
    if best_rmse > 500000:
        print(f"\n⚠️  RMSE ({best_rmse:,.2f}) больше целевого значения 500,000")
        print("Попробуем улучшить модель...")
        
        # Попробуем более сложную модель с другими параметрами
        print("\nОбучение улучшенной модели...")
        improved_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        improved_model.fit(X_train, y_train)
        y_pred_improved = improved_model.predict(X_test)
        rmse_improved = get_rmse(y_test, y_pred_improved)
        
        print(f"RMSE улучшенной модели: {rmse_improved:,.2f}")
        
        if rmse_improved < best_rmse:
            best_model = improved_model
            best_rmse = rmse_improved
            best_model_name = "Improved XGBoost"
    
    # Предсказания на тестовом датасете
    print("\nПредобработка тестового датасета...")
    X_test_final, _, _ = preprocess_data(
        test_data, 
        is_train=False, 
        label_encoders=label_encoders, 
        scaler=scaler
    )
    
    print("Делаем предсказания на тестовом датасете...")
    predictions = best_model.predict(X_test_final)
    
    # Создаем submission файл
    print("\nСоздание submission файла...")
    submission = pd.DataFrame({
        'index': np.arange(len(predictions)),
        'price': predictions.astype(int)
    })
    
    output_path = '/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"Submission файл сохранен: {output_path}")
    print(f"Размер submission: {submission.shape}")
    print(f"\nПервые 10 предсказаний:")
    print(submission.head(10))
    
    # Финальная оценка на полном обучающем датасете (для информации)
    print("\n" + "="*60)
    print("Финальная оценка на полном обучающем датасете:")
    y_train_pred = best_model.predict(X)
    rmse_full = get_rmse(y, y_train_pred)
    print(f"RMSE на полном датасете: {rmse_full:,.2f}")
    print("="*60)
    
    return best_model, best_rmse, label_encoders, scaler

if __name__ == '__main__':
    train_and_evaluate()

