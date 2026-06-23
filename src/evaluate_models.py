import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# настройка шрифтов для графиков
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8

# создание папки для графиков
os.makedirs('../plots/evaluation', exist_ok=True)

# загрузка тестовых данных
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

# загрузка обученных моделей
lr = joblib.load('../models/linear_regression_model.pkl')
rf = joblib.load('../models/random_forest_model.pkl')
xgb = joblib.load('../models/xgboost_model.pkl')
nn = load_model('models/neural_network_model.h5')

# получение предсказаний
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
y_pred_nn = nn.predict(X_test)

models = ['Линейная регрессия', 'Случайный лес', 'XGBoost', 'Нейронная сеть']
predictions = [y_pred_lr, y_pred_rf, y_pred_xgb, y_pred_nn]

# сбор всех метрик качества
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

# гистограмма сравнения R² с подписями
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width / 2, df_results['R² (депрессия)'], width, label='Депрессия', color='steelblue')
bars2 = ax.bar(x + width / 2, df_results['R² (тревожность)'], width, label='Тревожность', color='coral')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Модель')
ax.set_ylabel('R²')
ax.set_title('Сравнение коэффициента детерминации')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('plots/evaluation/comparison_r2.png', dpi=150)
plt.show()

# гистограмма сравнения MSE с подписями
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, df_results['MSE (депрессия)'], width, label='Депрессия', color='steelblue')
bars2 = ax.bar(x + width / 2, df_results['MSE (тревожность)'], width, label='Тревожность', color='coral')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Модель')
ax.set_ylabel('MSE')
ax.set_title('Сравнение среднеквадратичной ошибки (MSE) — чем меньше, тем лучше')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
plt.tight_layout()
plt.savefig('plots/evaluation/comparison_mse.png', dpi=150)
plt.show()

# гистограмма сравнения MAE с подписями
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, df_results['MAE (депрессия)'], width, label='Депрессия', color='steelblue')
bars2 = ax.bar(x + width / 2, df_results['MAE (тревожность)'], width, label='Тревожность', color='coral')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Модель')
ax.set_ylabel('MAE')
ax.set_title('Сравнение средней абсолютной ошибки (MAE) — чем меньше, тем лучше')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
plt.tight_layout()
plt.savefig('plots/evaluation/comparison_mae.png', dpi=150)
plt.show()

# точечные графики предсказанных vs реальных значений (без R² в заголовках)
for name, pred in zip(models, predictions):
    fig, axes = plt.subplots(2, 1, figsize=(5, 7))

    axes[0].scatter(y_test['depression_probability'], pred[:, 0], alpha=0.4, s=8)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=0.8)
    axes[0].set_xlabel('Реальные значения', fontsize=7)
    axes[0].set_ylabel('Предсказанные значения', fontsize=7)
    axes[0].set_title(f'{name} — депрессия', fontsize=8)
    axes[0].tick_params(labelsize=6)

    axes[1].scatter(y_test['anxiety_probability'], pred[:, 1], alpha=0.4, s=8, color='orange')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=0.8)
    axes[1].set_xlabel('Реальные значения', fontsize=7)
    axes[1].set_ylabel('Предсказанные значения', fontsize=7)
    axes[1].set_title(f'{name} — тревожность', fontsize=8)
    axes[1].tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(f'plots/evaluation/scatter_{name.replace(" ", "_")}.png', dpi=150)
    plt.show()

# важность признаков для XGBoost
X_train = pd.read_csv('../data/processed/X_train.csv')
feature_names = X_train.columns
importance = xgb.feature_importances_
imp_df = pd.DataFrame({'Признак': feature_names, 'Важность': importance}).sort_values('Важность', ascending=False)

plt.figure(figsize=(9, 5))
plt.barh(imp_df['Признак'][:8], imp_df['Важность'][:8], color='teal')
plt.xlabel('Важность', fontsize=8)
plt.title('Важность признаков (XGBoost)', fontsize=9)
plt.gca().invert_yaxis()
plt.tick_params(labelsize=7)
for i, (idx, row) in enumerate(imp_df.head(8).iterrows()):
    plt.text(row['Важность'] + 0.005, i, f'{row["Важность"]:.3f}', va='center', fontsize=7)
plt.tight_layout()
plt.savefig('plots/evaluation/feature_importance.png', dpi=150)
plt.show()

# анализ остатков для XGBoost
best_pred = y_pred_xgb
residuals_dep = y_test['depression_probability'] - best_pred[:, 0]
residuals_anx = y_test['anxiety_probability'] - best_pred[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(residuals_dep, bins=50, edgecolor='black')
axes[0].set_xlabel('Остаток', fontsize=8)
axes[0].set_ylabel('Частота', fontsize=8)
axes[0].set_title('Депрессия: остатки (XGBoost)', fontsize=9)
axes[0].axvline(x=0, color='r', linestyle='--')
axes[0].tick_params(labelsize=7)
axes[0].text(0.05, axes[0].get_ylim()[1] * 0.85,
             f'среднее = {residuals_dep.mean():.4f}\nстд = {residuals_dep.std():.4f}',
             transform=axes[0].transData, fontsize=7, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[1].hist(residuals_anx, bins=50, edgecolor='black', color='orange')
axes[1].set_xlabel('Остаток', fontsize=8)
axes[1].set_ylabel('Частота', fontsize=8)
axes[1].set_title('Тревожность: остатки (XGBoost)', fontsize=9)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].tick_params(labelsize=7)
axes[1].text(0.05, axes[1].get_ylim()[1] * 0.85,
             f'среднее = {residuals_anx.mean():.4f}\nстд = {residuals_anx.std():.4f}',
             transform=axes[1].transData, fontsize=7, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig('plots/evaluation/residuals.png', dpi=150)
plt.show()