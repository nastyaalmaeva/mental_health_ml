import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# загрузка моделей
lr = joblib.load('linear_regression_model.pkl')
rf = joblib.load('random_forest_model.pkl')
xgb = joblib.load('xgboost_model.pkl')
nn = load_model('neural_network_model.h5')

# предсказания
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
y_pred_nn = nn.predict(X_test)

models = ['Линейная регрессия', 'Случайный лес', 'XGBoost', 'Нейронная сеть']
predictions = [y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_nn]

# сбор метрик R²
results = []
for name, pred in zip(models, predictions):
    results.append({
        'Модель': name,
        'R² (депрессия)': round(r2_score(y_test['depression_probability'], pred[:, 0]), 4),
        'R² (тревожность)': round(r2_score(y_test['anxiety_probability'], pred[:, 1]), 4),
        'MSE (депрессия)': round(mean_squared_error(y_test['depression_probability'], pred[:, 0]), 4),
        'MAE (депрессия)': round(mean_absolute_error(y_test['depression_probability'], pred[:, 0]), 4),
        'MSE (тревожность)': round(mean_squared_error(y_test['anxiety_probability'], pred[:, 1]), 4),
        'MAE (тревожность)': round(mean_absolute_error(y_test['anxiety_probability'], pred[:, 1]), 4)
    })

df_results = pd.DataFrame(results)
print('=' * 70)
print('СРАВНЕНИЕ МОДЕЛЕЙ')
print('=' * 70)
print(df_results.to_string(index=False))

# график 1: гистограмма сравнения R²
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
width = 0.35
ax.bar(x - width / 2, df_results['R² (депрессия)'], width, label='Депрессия', color='steelblue')
ax.bar(x + width / 2, df_results['R² (тревожность)'], width, label='Тревожность', color='coral')
ax.set_xlabel('Модель')
ax.set_ylabel('R²')
ax.set_title('Сравнение коэффициента детерминации')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('comparison_r2.png', dpi=150)
plt.show()

# график 2: точечные графики для всех моделей (депрессия и тревожность)
for name, pred in zip(models, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # депрессия
    axes[0].scatter(y_test['depression_probability'], pred[:, 0], alpha=0.5, s=10)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[0].set_xlabel('Реальные значения')
    axes[0].set_ylabel('Предсказанные значения')
    axes[0].set_title(f'{name} — депрессия (R²={r2_score(y_test["depression_probability"], pred[:, 0]):.4f})')

    # тревожность
    axes[1].scatter(y_test['anxiety_probability'], pred[:, 1], alpha=0.5, s=10, color='orange')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[1].set_xlabel('Реальные значения')
    axes[1].set_ylabel('Предсказанные значения')
    axes[1].set_title(f'{name} — тревожность (R²={r2_score(y_test["anxiety_probability"], pred[:, 1]):.4f})')

    plt.tight_layout()
    plt.savefig(f'scatter_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()

# график 3: важность признаков (XGBoost)
feature_names = pd.read_csv('X_train.csv').columns
importance = xgb.feature_importances_
imp_df = pd.DataFrame({'Признак': feature_names, 'Важность': importance}).sort_values('Важность', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(imp_df['Признак'][:8], imp_df['Важность'][:8], color='teal')
plt.xlabel('Важность')
plt.title('Важность признаков (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# график 4: остатки для лучшей модели (XGBoost)
best_pred = y_pred_xgb
best_name = 'XGBoost'
residuals_dep = y_test['depression_probability'] - best_pred[:, 0]
residuals_anx = y_test['anxiety_probability'] - best_pred[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals_dep, bins=50, edgecolor='black')
axes[0].set_xlabel('Остаток')
axes[0].set_ylabel('Частота')
axes[0].set_title(f'Депрессия: остатки ({best_name})')
axes[0].axvline(x=0, color='r', linestyle='--')
axes[1].hist(residuals_anx, bins=50, edgecolor='black', color='orange')
axes[1].set_xlabel('Остаток')
axes[1].set_ylabel('Частота')
axes[1].set_title(f'Тревожность: остатки ({best_name})')
axes[1].axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
plt.show()