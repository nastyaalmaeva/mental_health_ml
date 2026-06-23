import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

os.makedirs('../models', exist_ok=True)

X_train = pd.read_csv('../data/processed/X_train.csv')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

model = RandomForestRegressor(
    n_estimators=100,
    max_features='sqrt',
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

joblib.dump(model, '../models/random_forest_model.pkl')
print('модель сохранена: models/random_forest_model.pkl')

y_pred = model.predict(X_test)
y_pred_dep = y_pred[:, 0]
y_pred_anx = y_pred[:, 1]

print(f'депрессия: MSE={mean_squared_error(y_test["depression_probability"], y_pred_dep):.4f}, MAE={mean_absolute_error(y_test["depression_probability"], y_pred_dep):.4f}, R²={r2_score(y_test["depression_probability"], y_pred_dep):.4f}')
print(f'тревожность: MSE={mean_squared_error(y_test["anxiety_probability"], y_pred_anx):.4f}, MAE={mean_absolute_error(y_test["anxiety_probability"], y_pred_anx):.4f}, R²={r2_score(y_test["anxiety_probability"], y_pred_anx):.4f}')

importance = pd.DataFrame({
    'признак': X_train.columns,
    'важность': model.feature_importances_
}).sort_values('важность', ascending=False)
print('\nважность признаков (топ-5):')
print(importance.head(5).to_string(index=False))