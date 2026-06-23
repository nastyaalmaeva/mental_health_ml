import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# настройка размеров шрифтов для читаемых графиков
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8

# загрузка данных
df = pd.read_csv('data\raw\mental_health.csv')

# первичная информация о датасете
print('размер:', df.shape)
print('\nпропуски:\n', df.isnull().sum())
print('\nстатистика числовых признаков:\n', df.describe())

# список числовых признаков
numeric_cols = ['age', 'stress_level', 'sleep_duration', 'physical_activity_days',
                'social_support_level', 'productivity_level']

# гистограммы распределений числовых признаков
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black')
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig('hist_numeric.png', dpi=150)
plt.show()

# гистограммы целевых переменных
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
df['depression_probability'].hist(ax=axes[0], bins=30, edgecolor='black')
axes[0].set_title('депрессия')
df['anxiety_probability'].hist(ax=axes[1], bins=30, edgecolor='black', color='orange')
axes[1].set_title('тревожность')
plt.tight_layout()
plt.savefig('targets_hist.png', dpi=150)
plt.show()

# список категориальных признаков
categorical_cols = ['gender', 'employment_status', 'work_conditions',
                    'mental_health_history', 'treatment_history']

# столбчатые диаграммы для категориальных признаков
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('categorical_counts.png', dpi=150)
plt.show()

# временное кодирование категорий для расчёта корреляций
df_encoded = df.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# полная корреляционная матрица
all_vars = numeric_cols + categorical_cols + ['depression_probability', 'anxiety_probability']
corr_matrix = df_encoded[all_vars].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'size': 6})
plt.title('корреляционная матрица')
plt.tight_layout()
plt.savefig('corr_matrix.png', dpi=150)
plt.show()

# вывод корреляций с целевыми переменными
print('\nкорреляции с целевыми:')
print(corr_matrix[['depression_probability', 'anxiety_probability']].round(3))