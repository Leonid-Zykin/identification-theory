#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

def remove_multicollinearity(X, threshold=0.95):
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    if to_drop:
        X = X.drop(columns=to_drop)
    
    return X, to_drop

def preprocess_data(df, is_train=True, onehot_encoder=None, scaler=None, 
                    columns_to_drop=None, feature_names=None):
    df = df.copy()
    
    df['area_ratio'] = df['kitchen_area'] / (df['total_area'] + 1e-6)
    df['bath_ratio'] = df['bath_area'] / (df['total_area'] + 1e-6)
    df['other_ratio'] = df['other_area'] / (df['total_area'] + 1e-6)
    df['extra_area_ratio'] = df['extra_area'] / (df['total_area'] + 1e-6)
    df['floor_ratio'] = df['floor'] / (df['floor_max'] + 1e-6)
    df['age'] = 2025 - df['year']
    df['area_per_room'] = df['total_area'] / (df['rooms_count'] + 1)
    df['bath_per_bathroom'] = df['bath_area'] / (df['bath_count'] + 1e-6)
    df['kitchen_per_total'] = df['kitchen_area'] / (df['total_area'] + 1e-6)
    
    df['total_rooms'] = df['rooms_count'] + df['bath_count']
    df['area_per_total_room'] = df['total_area'] / (df['total_rooms'] + 1)
    
    df['total_area_sq'] = df['total_area'] ** 2
    df['total_area_cub'] = df['total_area'] ** 3
    df['age_sq'] = df['age'] ** 2
    df['rooms_area_interaction'] = df['rooms_count'] * df['total_area']
    df['floor_area_interaction'] = df['floor'] * df['total_area']
    df['ceil_height_area'] = df['ceil_height'] * df['total_area']
    df['rooms_floor_interaction'] = df['rooms_count'] * df['floor']
    df['year_area_interaction'] = df['year'] * df['total_area']
    
    df['log_total_area'] = np.log1p(df['total_area'])
    df['log_kitchen_area'] = np.log1p(df['kitchen_area'])
    df['log_extra_area'] = np.log1p(df['extra_area'] + 1)
    
    df['kitchen_bath_ratio'] = df['kitchen_area'] / (df['bath_area'] + 1e-6)
    df['total_extra_ratio'] = (df['total_area'] + df['extra_area']) / (df['total_area'] + 1e-6)
    df['floor_ratio_sq'] = df['floor_ratio'] ** 2
    df['area_per_room_sq'] = df['area_per_room'] ** 2
    
    numeric_cols = [
        'kitchen_area', 'bath_area', 'other_area', 'extra_area', 
        'extra_area_count', 'year', 'ceil_height', 'floor_max', 
        'floor', 'total_area', 'bath_count', 'rooms_count',
        'area_ratio', 'bath_ratio', 'other_ratio', 'extra_area_ratio', 
        'floor_ratio', 'age', 'area_per_room', 'bath_per_bathroom',
        'kitchen_per_total', 'area_per_total_room',
        'total_area_sq', 'total_area_cub', 'age_sq',
        'rooms_area_interaction', 'floor_area_interaction', 
        'ceil_height_area', 'rooms_floor_interaction', 'year_area_interaction',
        'log_total_area', 'log_kitchen_area', 'log_extra_area',
        'kitchen_bath_ratio', 'total_extra_ratio', 
        'floor_ratio_sq', 'area_per_room_sq'
    ]
    
    categorical_cols = ['gas', 'hot_water', 'central_heating', 
                       'extra_area_type_name', 'district_name']
    
    X_numeric = df[numeric_cols].copy()
    
    if is_train:
        onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_categorical = onehot_encoder.fit_transform(df[categorical_cols])
        feature_names_cat = onehot_encoder.get_feature_names_out(categorical_cols)
    else:
        if onehot_encoder is None:
            raise ValueError("onehot_encoder must be provided for test data")
        X_categorical = onehot_encoder.transform(df[categorical_cols])
        feature_names_cat = onehot_encoder.get_feature_names_out(categorical_cols)
    
    X_categorical_df = pd.DataFrame(
        X_categorical, 
        columns=feature_names_cat,
        index=df.index
    )
    
    X = pd.concat([X_numeric, X_categorical_df], axis=1)
    
    district_cols = [col for col in X_categorical_df.columns if 'district_name' in col]
    important_numeric = ['total_area', 'rooms_count', 'year', 'floor', 'ceil_height']
    
    if is_train:
        for district_col in district_cols:
            for num_col in important_numeric:
                if num_col in X_numeric.columns:
                    interaction_name = f'{district_col}_x_{num_col}'
                    X[interaction_name] = X_categorical_df[district_col] * X_numeric[num_col]
    else:
        for district_col in district_cols:
            for num_col in important_numeric:
                if num_col in X_numeric.columns:
                    interaction_name = f'{district_col}_x_{num_col}'
                    if interaction_name in feature_names:
                        X[interaction_name] = X_categorical_df[district_col] * X_numeric[num_col]
    
    if is_train:
        variance_selector = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(
            variance_selector.fit_transform(X),
            columns=X.columns[variance_selector.get_support()],
            index=X.index
        )
    else:
        if feature_names is None:
            raise ValueError("feature_names must be provided for test data")
        X = X[feature_names]
    
    if is_train:
        X, columns_to_drop = remove_multicollinearity(X, threshold=0.95)
        feature_names = X.columns.tolist()
    else:
        if columns_to_drop:
            X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
        X = X[feature_names]
    
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
        return X_scaled, y, onehot_encoder, scaler, columns_to_drop, feature_names
    else:
        return X_scaled, onehot_encoder, scaler, columns_to_drop, feature_names

# Определяем пути к данным (для Kaggle и локального запуска)
import os
import glob

def find_file(filename, search_dirs=['/kaggle/input', '.', './']):
    """Ищет файл в указанных директориях и поддиректориях"""
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Ищем в корне директории
            path = os.path.join(search_dir, filename)
            if os.path.exists(path):
                return path
            # Ищем рекурсивно во всех поддиректориях
            pattern = os.path.join(search_dir, '**', filename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
    return None

# Ищем файлы данных
train_path = find_file('data.csv')
test_path = find_file('test.csv')

# Если не нашли, используем локальные пути (для тестирования)
if train_path is None:
    train_path = '/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/data.csv'
if test_path is None:
    test_path = '/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/test.csv'

# Путь для сохранения submission
if os.path.exists('/kaggle/working'):
    output_path = '/kaggle/working/submission.csv'
else:
    output_path = '/home/leonidas/projects/itmo/identification-theory/lab3/Archive2025/submission.csv'

print("Загрузка данных...")
data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Предобработка обучающих данных...")
X, y, onehot_encoder, scaler, columns_to_drop, feature_names = preprocess_data(data, is_train=True)

print("Обучение модели линейной регрессии...")
model = LinearRegression()
model.fit(X, y)

print("Предобработка тестовых данных...")
X_test, _, _, _, _ = preprocess_data(
    test_data, 
    is_train=False, 
    onehot_encoder=onehot_encoder, 
    scaler=scaler,
    columns_to_drop=columns_to_drop,
    feature_names=feature_names
)

print("Делаем предсказания...")
predictions = model.predict(X_test)

print("Создание submission файла...")
submission = pd.DataFrame({
    'index': np.arange(len(predictions)),
    'price': predictions.astype(int)
})

submission.to_csv(output_path, index=False)
print(f"Submission файл сохранен: {output_path}")
