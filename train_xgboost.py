import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, 'xgboost_model.pkl')
print('модель сохранена: xgboost_model.pkl')

y_pred = model.predict(X_test)
y_pred_dep = y_pred[:, 0]
y_pred_anx = y_pred[:, 1]

print('=' * 50)
print('XGBOOST')
print('=' * 50)
print(f'депрессия: MSE={mean_squared_error(y_test["depression_probability"], y_pred_dep):.4f}, MAE={mean_absolute_error(y_test["depression_probability"], y_pred_dep):.4f}, R²={r2_score(y_test["depression_probability"], y_pred_dep):.4f}')
print(f'тревожность: MSE={mean_squared_error(y_test["anxiety_probability"], y_pred_anx):.4f}, MAE={mean_absolute_error(y_test["anxiety_probability"], y_pred_anx):.4f}, R²={r2_score(y_test["anxiety_probability"], y_pred_anx):.4f}')

# важность признаков
importance = pd.DataFrame({
    'признак': X_train.columns,
    'важность': model.feature_importances_
}).sort_values('важность', ascending=False)
print('\nважность признаков (топ-5):')
print(importance.head(5).to_string(index=False))