import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# загрузка выборок
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# обучение
model = LinearRegression()
model.fit(X_train, y_train)

# сохранение модели
joblib.dump(model, 'linear_regression_model.pkl')
print('модель сохранена: linear_regression_model.pkl')

# прогноз на тесте
y_pred = model.predict(X_test)

# метрики
y_pred_dep = y_pred[:, 0]
y_pred_anx = y_pred[:, 1]

print('=' * 50)
print('ЛИНЕЙНАЯ РЕГРЕССИЯ')
print('=' * 50)
print(f'депрессия: MSE={mean_squared_error(y_test["depression_probability"], y_pred_dep):.4f}, MAE={mean_absolute_error(y_test["depression_probability"], y_pred_dep):.4f}, R²={r2_score(y_test["depression_probability"], y_pred_dep):.4f}')
print(f'тревожность: MSE={mean_squared_error(y_test["anxiety_probability"], y_pred_anx):.4f}, MAE={mean_absolute_error(y_test["anxiety_probability"], y_pred_anx):.4f}, R²={r2_score(y_test["anxiety_probability"], y_pred_anx):.4f}')