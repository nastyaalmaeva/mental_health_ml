import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

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

# сбор R²
r2_dep = []
r2_anx = []
for pred in predictions:
    r2_dep.append(r2_score(y_test['depression_probability'], pred[:, 0]))
    r2_anx.append(r2_score(y_test['anxiety_probability'], pred[:, 1]))

# таблица результатов
print('=' * 50)
print('СРАВНЕНИЕ МОДЕЛЕЙ (R²)')
print('=' * 50)
print(f"{'Модель':<20} {'Депрессия':>10} {'Тревожность':>12}")
print('-' * 50)
for i, name in enumerate(models):
    print(f"{name:<20} {r2_dep[i]:>10.4f} {r2_anx[i]:>12.4f}")

# график 1: группированные столбцы
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, r2_dep, width, label='Депрессия', color='steelblue')
bars2 = ax.bar(x + width/2, r2_anx, width, label='Тревожность', color='coral')

# подписи значений на столбцах
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Модель')
ax.set_ylabel('R²')
ax.set_title('Сравнение коэффициента детерминации')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('comparison_r2.png', dpi=150)
plt.show()

# график 2: точечный график (все модели на одном поле)
fig, ax = plt.subplots(figsize=(8, 6))
offset = 0
for i, name in enumerate(models):
    offset += 0.02
    ax.scatter(r2_dep[i], r2_anx[i], s=150, label=name, alpha=0.8)
    ax.annotate(name, (r2_dep[i] + 0.01, r2_anx[i] + 0.01), fontsize=8)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='идеальная модель')
ax.set_xlabel('R² (депрессия)')
ax.set_ylabel('R² (тревожность)')
ax.set_title('Качество моделей: депрессия vs тревожность')
ax.set_xlim(0.65, 1)
ax.set_ylim(0.65, 1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('scatter_r2_comparison.png', dpi=150)
plt.show()

# график 3: важность признаков (XGBoost)
feature_names = pd.read_csv('X_train.csv').columns
importance = xgb.feature_importances_
imp_df = pd.DataFrame({'Признак': feature_names, 'Важность': importance}).sort_values('Важность', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(imp_df['Признак'], imp_df['Важность'], color='teal')
plt.xlabel('Важность')
plt.title('Важность признаков (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()