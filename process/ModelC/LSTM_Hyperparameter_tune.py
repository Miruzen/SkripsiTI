#!/usr/bin/env python3
"""
lstm_hyperparameter_tuning.py
Cari kombinasi hyperparameter terbaik untuk LSTM menggunakan Bayesian Optimization.
Output:
    - best_model_lstm.h5
    - best_params.json
    - tuning_results.csv
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error
import keras_tuner as kt

# ==========================================================
# CONFIG
# ==========================================================
DATA_FILE = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC\processed_data.npz"
OUTPUT_DIR = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC\HyperTuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 60
LOOKBACK = 7  # harus sama seperti data preprocessing
VAL_SPLIT = 0.1

# ==========================================================
# LOAD DATA
# ==========================================================
print(f"📂 Loading dataset from: {DATA_FILE}")
data = np.load(DATA_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
print(f"✅ Loaded! X_train: {X_train.shape}, y_train: {y_train.shape}")

# ==========================================================
# MODEL BUILDER FUNCTION
# ==========================================================
def build_model(hp):
    model = Sequential()
    # Layer 1
    model.add(LSTM(
        units=hp.Choice("lstm_units_1", [32, 64, 128]),
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Choice("dropout_1", [0.1, 0.2, 0.3, 0.4])))

    # Layer 2
    model.add(LSTM(
        units=hp.Choice("lstm_units_2", [16, 32, 64]),
        return_sequences=False
    ))
    model.add(Dropout(hp.Choice("dropout_2", [0.1, 0.2, 0.3])))

    # Dense Layer
    model.add(Dense(hp.Choice("dense_units", [8, 16, 32]), activation="relu"))
    model.add(Dense(1))  # output layer

    # Optimizer
    model.compile(
        optimizer=hp.Choice("optimizer", ["adam"]),
        loss="mse"
    )
    return model

# ==========================================================
# TUNER SETUP (Bayesian Optimization)
# ==========================================================
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=15,          # jumlah kombinasi parameter yang diuji
    directory=OUTPUT_DIR,
    project_name="LSTM_Tuning"
)

# ==========================================================
# TRAINING / SEARCHING
# ==========================================================
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

print("\n🚀 Starting Bayesian hyperparameter tuning...")
tuner.search(
    X_train, y_train,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================================
# BEST MODEL & PARAMETER
# ==========================================================
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

# Simpan parameter terbaik
best_params_dict = best_hyperparams.values
params_path = os.path.join(OUTPUT_DIR, "best_params.json")
with open(params_path, "w") as f:
    json.dump(best_params_dict, f, indent=4)
print(f"💾 Best hyperparameters saved to: {params_path}")

# Simpan model terbaik
model_path = os.path.join(OUTPUT_DIR, "best_model_lstm.h5")
best_model.save(model_path)
print(f"✅ Best model saved to: {model_path}")

# ==========================================================
# EVALUASI MODEL TERBAIK
# ==========================================================
y_pred = best_model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"\n📊 Final Test MAPE (best model): {mape:.4f}%")

# Simpan hasil evaluasi
results = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})
results["Error_%"] = abs((results["Actual"] - results["Predicted"]) / results["Actual"]) * 100
results.to_excel(os.path.join(OUTPUT_DIR, "best_model_results.xlsx"), index=False)

# ==========================================================
# VISUALISASI HASIL
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(results["Actual"].values, label="Actual", color="black")
plt.plot(results["Predicted"].values, label="Predicted", color="orange", linestyle="--")
plt.title(f"Best LSTM Prediction (MAPE: {mape:.2f}%)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Close Price")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "best_model_plot.png")
plt.savefig(plot_path, dpi=200)
plt.show()
print(f"📈 Visualization saved to: {plot_path}")

print("\n🏁 Hyperparameter tuning completed successfully!")
