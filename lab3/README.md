# Лабораторная работа №3: Предсказание цен квартир

## Задача
Предсказать цены квартир на основе характеристик недвижимости. Целевая метрика: **RMSE < 500,000**.

## Решение

### Предобработка данных
1. **Feature Engineering:**
   - `area_ratio` = kitchen_area / total_area
   - `bath_ratio` = bath_area / total_area
   - `extra_area_ratio` = extra_area / total_area
   - `floor_ratio` = floor / floor_max
   - `age` = 2025 - year (возраст квартиры)
   - `area_per_room` = total_area / (rooms_count + 1)
   - `bath_per_bathroom` = bath_area / (bath_count + 1)

2. **Кодирование категориальных признаков:**
   - `gas`, `hot_water`, `central_heating` (Yes/No) → Label Encoding
   - `extra_area_type_name` (balcony/loggia) → Label Encoding
   - `district_name` (районы) → Label Encoding

3. **Масштабирование:** StandardScaler для числовых признаков

### Разделение данных
- Обучающая выборка: 70% (70,000 записей)
- Валидационная выборка: 30% (30,000 записей)
- Используется случайная перестановка с seed=42 для воспроизводимости

### Модели
Протестированы следующие модели:
- **GradientBoostingRegressor** (лучшая) - RMSE: **189,061.17**
- XGBoost - RMSE: 199,799.29
- LightGBM - RMSE: 205,007.97
- RandomForest - RMSE: 328,605.06

### Параметры лучшей модели (GradientBoosting)
```python
GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```

## Результаты

✅ **RMSE на валидации: 189,061.17** (цель: < 500,000)

✅ Submission файл создан: `Archive2025/submission.csv`

Формат submission:
- Колонки: `index`, `price`
- Количество записей: 100,000
- Диапазон цен: от 4,746,345 до 43,495,336 рублей

## Использование

```bash
# Активировать виртуальное окружение
source venv/bin/activate

# Запустить обучение модели
python train_model.py
```

## Файлы
- `train_model.py` - основной скрипт для обучения модели
- `Archive2025/data.csv` - обучающий датасет (100,000 записей)
- `Archive2025/test.csv` - тестовый датасет (100,000 записей)
- `Archive2025/submission.csv` - файл с предсказаниями для отправки

## Формула RMSE
```
RMSE = sqrt(mean((price_actual - price_predicted)^2))
```


