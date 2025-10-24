#!/usr/bin/env python3
"""
prepare_lstm_data.py
Mempersiapkan data untuk Stacked LSTM (FinBERT-LSIMF pipeline)

Input  : mood_series.xlsx
Output : processed_data.xlsx, processed_data.npz
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# CONFIG
# ==========================================================
input_file = r"D:\Skripsi\Data\Kode Ekstrak\output\mood_series.xlsx"
output_xlsx = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC\processed_data.xlsx"
output_npz = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelC\processed_data.npz"
lookback = 10  # jendela waktu untuk LSTM

# ==========================================================
# LOAD DATA
# ==========================================================
print("📂 Loading mood series data...")
df = pd.read_excel(input_file)

df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ==========================================================
# PILIH FITUR
# ==========================================================
features = [
    "mood_score",
    "t_pos", "t_neg",
    "c_pos", "c_neg",
    "norm_close", "norm_ema20", "norm_ema50"
]

df = df[["date"] + features].dropna()
print(f"✅ Fitur digunakan: {features}")
print(f"📏 Data shape awal: {df.shape}")

# ==========================================================
# SCALING NILAI FITUR (jika diperlukan)
# ==========================================================
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])
scaled_df = pd.DataFrame(scaled_features, columns=features)
scaled_df["date"] = df["date"]

# ==========================================================
# BENTUK SEQUENCE UNTUK LSTM
# ==========================================================
X, y, dates = [], [], []

for i in range(len(scaled_df) - lookback):
    window = scaled_df[features].iloc[i:i + lookback].values
    target = scaled_df["norm_close"].iloc[i + lookback]  
    X.append(window)
    y.append(target)
    dates.append(scaled_df["date"].iloc[i + lookback])

X = np.array(X)
y = np.array(y)
print(f"📈 Bentuk data untuk LSTM: X={X.shape}, y={y.shape}")

# ==========================================================
# SPLIT TRAIN/TEST
# ==========================================================
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==========================================================
# SIMPAN HASIL .NPZ
# ==========================================================
np.savez(output_npz, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print(f"💾 Data LSTM disimpan ke: {output_npz}")

# ==========================================================
# SIMPAN VERSI EXCEL UNTUK INSPEKSI MANUAL
# ==========================================================
out_df = pd.DataFrame({
    "date": dates,
    "target_norm_close": y
})
out_df.to_excel(output_xlsx, index=False, engine="openpyxl")

print(f"📘 Data tabular disimpan ke: {output_xlsx}")
print("✅ Tahap persiapan data LSTM selesai!")
