#!/usr/bin/env python3
"""
ModelB Generator with Normalized Close (Full Merge)
---------------------------------------------------
Input  : data/dataeurusd.xlsx  (kolom: date, close)
Outputs:
    - output/ModelB/ModelBtrain_*.xlsx & ModelBtest_*.xlsx
    - output/ModelB/ModelBsummary.xlsx (MAPE ringkasan)
    - output/ModelB/ModelBfinal.xlsx   (TRAIN + TEST digabung + normalisasi)
    - output/ModelB/ModelBfinal_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# CONFIG
# ==============================================================
INPUT_FILE = r"data/dataeurusd.xlsx"
OUTPUT_DIR = r"output/ModelB/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================
# UTILITAS
# ==============================================================
def ema_manual(prices, span):
    ema = [np.nan] * len(prices)
    alpha = 2 / (span + 1)
    for i in range(len(prices)):
        if i < span - 1:
            ema[i] = np.nan
        elif i == span - 1:
            ema[i] = np.mean(prices[:span])
        else:
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def mape_manual(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / np.abs(actual[mask]))) * 100


def minmax_scale_manual(series):
    x_min, x_max = np.nanmin(series), np.nanmax(series)
    if x_max == x_min:
        return np.full(len(series), 0.5)
    return (series - x_min) / (x_max - x_min)


def split_data(df, ratio=None, by_date=False):
    """Split dataset menjadi train dan test"""
    if by_date:
        train_df = df[df["date"].dt.year <= 2023].copy()
        test_df = df[df["date"].dt.year == 2024].copy()
        label = "byDate"
    else:
        split_index = int(len(df) * ratio)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        label = str(int(ratio * 100))
    return train_df, test_df, label


# ==============================================================
# LOAD DATA
# ==============================================================
print(f"📂 Membaca data dari {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

if not {"date", "close"}.issubset(df.columns):
    raise ValueError("❌ File harus memiliki kolom 'date' dan 'close'.")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# ==============================================================
# PROSES SPLIT
# ==============================================================
splits = [
    {"ratio": 0.7, "by_date": False},
    {"ratio": 0.8, "by_date": False},
    {"ratio": None, "by_date": True},
]

summary = []

for s in splits:
    train_df, test_df, label = split_data(df, s["ratio"], s["by_date"])
    print(f"\n🔹 Memproses split: {label}")

    for subset, name in [(train_df, "TRAIN"), (test_df, "TEST")]:
        prices = subset["close"].values
        subset["EMA20"] = ema_manual(prices, 20)
        subset["EMA50"] = ema_manual(prices, 50)

        subset["EMA20_shifted"] = subset["EMA20"].shift(1)
        subset["EMA50_shifted"] = subset["EMA50"].shift(1)

        subset["MAPE20"] = np.abs((subset["close"] - subset["EMA20_shifted"]) / subset["close"]) * 100
        subset["MAPE50"] = np.abs((subset["close"] - subset["EMA50_shifted"]) / subset["close"]) * 100
        subset["MAPE_mean"] = subset[["MAPE20", "MAPE50"]].mean(axis=1)

        avg_mape20 = mape_manual(subset["close"], subset["EMA20_shifted"])
        avg_mape50 = mape_manual(subset["close"], subset["EMA50_shifted"])
        avg_mape_total = np.nanmean([avg_mape20, avg_mape50])

        print(f"📊 {name} ({label}) → MAPE20: {avg_mape20:.4f}% | MAPE50: {avg_mape50:.4f}% | Mean: {avg_mape_total:.4f}%")

        summary_row = pd.DataFrame({
            "date": ["AVERAGE MAPE SPLIT"],
            "close": [np.nan],
            "EMA20": [np.nan],
            "EMA50": [np.nan],
            "EMA20_shifted": [np.nan],
            "EMA50_shifted": [np.nan],
            "MAPE20": [avg_mape20],
            "MAPE50": [avg_mape50],
            "MAPE_mean": [avg_mape_total]
        })
        subset = pd.concat([subset, summary_row], ignore_index=True)

        out_path = os.path.join(OUTPUT_DIR, f"ModelB{name}_{label}.xlsx")
        subset.to_excel(out_path, index=False, engine="openpyxl")
        print(f"💾 Disimpan ke {out_path}")

    avg_mape_train = mape_manual(train_df["close"], train_df["EMA50_shifted"])
    avg_mape_test = mape_manual(test_df["close"], test_df["EMA50_shifted"])
    summary.append({
        "Split": label,
        "Train_MAPE": round(avg_mape_train, 4),
        "Test_MAPE": round(avg_mape_test, 4),
        "Mean_MAPE": round(np.nanmean([avg_mape_train, avg_mape_test]), 4)
    })


# ==============================================================
# PILIH SPLIT TERBAIK
# ==============================================================
summary_df = pd.DataFrame(summary)
best_row = summary_df.loc[summary_df["Mean_MAPE"].idxmin()]
best_split = best_row["Split"]
summary_df["Best_Model"] = ["✅" if s == best_split else "" for s in summary_df["Split"]]

print(f"\n🏁 Split terbaik berdasarkan rata-rata MAPE: {best_split}")


# ==============================================================
# GABUNGKAN TRAIN + TEST DARI SPLIT TERBAIK
# ==============================================================
train_path = os.path.join(OUTPUT_DIR, f"ModelBTRAIN_{best_split}.xlsx")
test_path = os.path.join(OUTPUT_DIR, f"ModelBTEST_{best_split}.xlsx")

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    raise FileNotFoundError(f"❌ File train/test untuk split terbaik tidak ditemukan ({best_split}).")

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# Gabungkan dan urutkan berdasarkan tanggal
full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df[~full_df["date"].astype(str).str.contains("AVERAGE", na=False)]  # hapus baris ringkasan
full_df["date"] = pd.to_datetime(full_df["date"], errors="coerce")
full_df = full_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ==============================================================
# NORMALISASI DAN SIMPAN FINAL
# ==============================================================
full_df["Norm_EMA20"] = minmax_scale_manual(full_df["EMA20"])
full_df["Norm_EMA50"] = minmax_scale_manual(full_df["EMA50"])
full_df["Norm_Close"] = minmax_scale_manual(full_df["close"])

final_cols = ["date", "close", "EMA20", "EMA50", "Norm_EMA20", "Norm_EMA50", "Norm_Close"]
final_df = full_df[final_cols].copy()

final_path = os.path.join(OUTPUT_DIR, "ModelBfinal.xlsx")
final_df.to_excel(final_path, index=False, engine="openpyxl")

print(f"\n✅ ModelBfinal berhasil dibuat dan mencakup TRAIN + TEST!")
print(f"📦 Disimpan di: {final_path}")
print(f"📏 Total baris data: {len(final_df)}")


# ==============================================================
# SIMPAN RINGKASAN DAN PLOT
# ==============================================================
summary_path = os.path.join(OUTPUT_DIR, "ModelBsummary.xlsx")
with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary_MAPE", index=False)
    full_df[["date", "close", "EMA20", "EMA50"]].to_excel(writer, sheet_name="Full_Data", index=False)
print(f"📘 Ringkasan disimpan ke {summary_path}")


# ==============================================================
# VISUALISASI
# ==============================================================
plt.figure(figsize=(12, 6))
plt.plot(full_df["date"], full_df["close"], label="Close", color="black", linewidth=1.5)
plt.plot(full_df["date"], full_df["EMA20"], label="EMA 20", color="blue", linestyle="--")
plt.plot(full_df["date"], full_df["EMA50"], label="EMA 50", color="red", linestyle="--")
plt.title(f"EUR/USD Close vs EMA20 & EMA50 (Gabungan Train + Test, Split: {best_split})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, "ModelBfinal_plot.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"📊 Grafik disimpan ke {plot_path}")
