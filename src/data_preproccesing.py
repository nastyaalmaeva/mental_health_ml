import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# создаём папки, если их нет
os.makedirs('../data/processed', exist_ok=True)

# загрузка данных из папки data/raw
df = pd.read_csv('../data/raw/mental_health.csv')

# оставляем только мужской и женский пол
df = df[df['gender'].isin(['Male', 'Female'])]

# список признаков (11 шт.) и целевых переменных (2 шт.)
feature_cols = ['age', 'gender', 'employment_status', 'work_conditions',
                'mental_health_history', 'treatment_history', 'stress_level',
                'sleep_duration', 'physical_activity_days', 'social_support_level',
                'productivity_level']
target_cols = ['depression_probability', 'anxiety_probability']

# кодирование категориальных признаков
categorical_cols = ['gender', 'employment_status', 'work_conditions',
                    'mental_health_history', 'treatment_history']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# разделение на признаки и целевые переменные
X = df[feature_cols].copy()
y = df[target_cols].copy()

# числовые признаки для стандартизации
numeric_cols = ['age', 'stress_level', 'sleep_duration', 'physical_activity_days',
                'social_support_level', 'productivity_level']

# стандартизация
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# разделение на train (80%) и test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# сохранение выборок в data/processed/
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)

print(f'размеры выборок:')
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
print(f'\nпример X_train (первые 3 строки):\n{X_train.head(3)}')
