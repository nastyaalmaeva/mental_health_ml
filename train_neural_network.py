import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, activation='swish'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='swish'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='swish'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='swish'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(2, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae', 'mse'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

model.save('neural_network_model.h5')
print('модель сохранена: neural_network_model.h5')

y_pred = model.predict(X_test)
y_pred_dep = y_pred[:, 0]
y_pred_anx = y_pred[:, 1]

print('=' * 50)
print('НЕЙРОННАЯ СЕТЬ')
print('=' * 50)
print(f'депрессия: MSE={mean_squared_error(y_test["depression_probability"], y_pred_dep):.4f}, MAE={mean_absolute_error(y_test["depression_probability"], y_pred_dep):.4f}, R²={r2_score(y_test["depression_probability"], y_pred_dep):.4f}')
print(f'тревожность: MSE={mean_squared_error(y_test["anxiety_probability"], y_pred_anx):.4f}, MAE={mean_absolute_error(y_test["anxiety_probability"], y_pred_anx):.4f}, R²={r2_score(y_test["anxiety_probability"], y_pred_anx):.4f}')