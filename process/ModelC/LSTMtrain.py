#!/usr/bin/env python3
"""
train_lstm_mape_full_dual.py
Train final LSTM model automatically using best hyperparameters from tuning.
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

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC"
DATA_FILE = os.path.join(BASE_DIR, "processed_data.npz")
PARAM_FILE = os.path.join(BASE_DIR, "best_params.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "final_model_training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 80
BATCH_SIZE = 32
LOOKBACK = 7

# ==========================================================
# LOAD DATA
# ==========================================================
print("📂 Loading processed dataset...")
data = np.load(DATA_FILE)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
dates_train = data["dates_train"]
dates_test = data["dates_test"]

print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}")

# ==========================================================
# LOAD BEST HYPERPARAMETERS
# ==========================================================
if not os.path.exists(PARAM_FILE):
    raise FileNotFoundError(f"❌ File {PARAM_FILE} not found. Run tuning first!")

with open(PARAM_FILE, "r") as f:
    params = json.load(f)

print("\n🧠 Loaded best hyperparameters:")
for k, v in params.items():
    print(f"   {k}: {v}")

# Default fallback jika ada parameter yang tidak lengkap
lstm1 = params.get("lstm_units_1", 64)
lstm2 = params.get("lstm_units_2", 32)
drop1 = params.get("dropout_1", 0.2)
drop2 = params.get("dropout_2", 0.2)
dense_units = params.get("dense_units", 16)
optimizer = params.get("optimizer", "adam")

# ==========================================================
# BUILD MODEL
# ==========================================================
model = Sequential([
    LSTM(lstm1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(drop1),
    LSTM(lstm2, return_sequences=False),
    Dropout(drop2),
    Dense(dense_units, activation="relu"),
    Dense(1)
])
model.compile(optimizer=optimizer, loss="mse")
model.summary()

# ==========================================================
# TRAINING
# ==========================================================
print("\n🚀 Training started...")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

print("\n✅ Training complete!")

# ==========================================================
# PREDICTION & MAPE
# ==========================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print(f"\n📊 Train MAPE: {mape_train:.2f}%")
print(f"📊 Test MAPE : {mape_test:.2f}%")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================
df_train = pd.DataFrame({
    "Date": dates_train,
    "Actual": y_train.flatten(),
    "Predicted": y_pred_train.flatten()
})
df_test = pd.DataFrame({
    "Date": dates_test,
    "Actual": y_test.flatten(),
    "Predicted": y_pred_test.flatten()
})

df_train.to_excel(os.path.join(OUTPUT_DIR, "LSTM_Train_Predictions.xlsx"), index=False)
df_test.to_excel(os.path.join(OUTPUT_DIR, "LSTM_Test_Predictions.xlsx"), index=False)

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(12, 6))
plt.plot(df_train["Date"], df_train["Actual"], label="Train Actual", color="gray", linewidth=1)
plt.plot(df_train["Date"], df_train["Predicted"], label="Train Predicted", color="green", linestyle="--")
plt.plot(df_test["Date"], df_test["Actual"], label="Test Actual", color="black", linewidth=1.3)
plt.plot(df_test["Date"], df_test["Predicted"], label="Test Predicted", color="orange", linestyle="--")
plt.title(f"LSTM Final Model — Train/Test (MAPE Train={mape_train:.2f}%, Test={mape_test:.2f}%)")
plt.xlabel("Date")
plt.ylabel("Normalized Close")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "LSTM_Final_Plot.png"), dpi=200)
plt.show()

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================
model.save(os.path.join(OUTPUT_DIR, "LSTM_Final_Model.h5"))
print(f"\n💾 Final model & results saved to: {OUTPUT_DIR}")
