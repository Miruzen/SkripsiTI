#!/usr/bin/env python3
"""
LSTManalisahasil.py
Analisis hasil prediksi dari model LSTM final (train_lstm_mape_full_dual.py)
Menampilkan metrik evaluasi dan visualisasi hasil prediksi Train & Test.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC\final_model_training"
TRAIN_FILE = os.path.join(BASE_DIR, "LSTM_Train_Predictions.xlsx")
TEST_FILE = os.path.join(BASE_DIR, "LSTM_Test_Predictions.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "LSTM_Evaluation_Summary.xlsx")

# ==========================================================
# LOAD DATA
# ==========================================================
print("📂 Loading LSTM prediction results...")

df_train = pd.read_excel(TRAIN_FILE)
df_test = pd.read_excel(TEST_FILE)

for df in [df_train, df_test]:
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(f"✅ Data loaded successfully!")
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# ==========================================================
# DEFINE METRIC FUNCTION
# ==========================================================
def compute_metrics(df):
    actual = df["actual"].values
    predicted = df["predicted"].values

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    corr = np.corrcoef(actual, predicted)[0, 1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "Correlation": corr
    }

# ==========================================================
# CALCULATE METRICS
# ==========================================================
metrics_train = compute_metrics(df_train)
metrics_test = compute_metrics(df_test)

summary = pd.DataFrame([
    {"Dataset": "Train", **metrics_train},
    {"Dataset": "Test", **metrics_test}
])

print("\n📊 Evaluation Metrics Summary:")
print(summary)

# ==========================================================
# SAVE SUMMARY
# ==========================================================
summary.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"\n💾 Evaluation summary saved to: {OUTPUT_FILE}")

# ==========================================================
# VISUALIZATION
# ==========================================================
plt.figure(figsize=(14, 6))
plt.suptitle("📈 LSTM Model Prediction vs Actual", fontsize=13, fontweight='bold')

# TRAIN
plt.subplot(1, 2, 1)
plt.plot(df_train["date"], df_train["actual"], label="Actual", color="black")
plt.plot(df_train["date"], df_train["predicted"], label="Predicted", color="green", linestyle="--")
plt.title(f"Train Data (MAPE: {metrics_train['MAPE (%)']:.2f}%)")
plt.xlabel("Date")
plt.ylabel("Normalized Close")
plt.legend()
plt.grid(alpha=0.4)

# TEST
plt.subplot(1, 2, 2)
plt.plot(df_test["date"], df_test["actual"], label="Actual", color="black")
plt.plot(df_test["date"], df_test["predicted"], label="Predicted", color="orange", linestyle="--")
plt.title(f"Test Data (MAPE: {metrics_test['MAPE (%)']:.2f}%)")
plt.xlabel("Date")
plt.ylabel("Normalized Close")
plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plot_path = os.path.join(BASE_DIR, "LSTM_Train_Test_Comparison.png")
plt.savefig(plot_path, dpi=200)
plt.show()

print(f"\n📊 Visualization saved to: {plot_path}")

# ==========================================================
# CORRELATION PLOT
# ==========================================================
plt.figure(figsize=(6, 6))
plt.scatter(df_test["actual"], df_test["predicted"], alpha=0.6, color="royalblue")
plt.title(f"Actual vs Predicted (Test Data)\nCorr = {metrics_test['Correlation']:.3f}")
plt.xlabel("Actual Normalized Close")
plt.ylabel("Predicted Normalized Close")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "LSTM_Correlation_Test.png"), dpi=200)
plt.show()

print("\n🏁 Analysis complete.")
