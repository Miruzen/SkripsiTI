#!/usr/bin/env python3
"""
Input  :
    - labeled_articles_final_FinBERT.xlsx
    - ModelBfinal.xlsx
Output :
    - mood_series.xlsx
"""

import pandas as pd
import os

# ==========================================================
# CONFIG
# ==========================================================
finbert_file = r"D:\Skripsi\Data\Kode Ekstrak\output\labeled_articles_final_FinBERT.xlsx"
modelb_file = r"D:\Skripsi\Data\Kode Ekstrak\output\ModelB\ModelBfinal.xlsx"
output_file = r"D:\Skripsi\Data\Kode Ekstrak\output\mood_series.xlsx"

# ==========================================================
# LOAD DATA
# ==========================================================
print("📂 Loading datasets...")
df_f = pd.read_excel(finbert_file)
df_b = pd.read_excel(modelb_file)

# ==========================================================
# NORMALIZE COLUMN NAMES
# ==========================================================
for df in [df_f, df_b]:
    df.columns = df.columns.str.lower().str.strip()

# ==========================================================
# FIX DATE COLUMN
# ==========================================================
for df in [df_f, df_b]:
    date_col = [c for c in df.columns if "date" in c][0]
    df.rename(columns={date_col: "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.normalize()  # remove time part

# ==========================================================
# AGGREGATE SENTIMENT (FinBERT)
# ==========================================================
print("🧠 Aggregating daily sentiment scores (FinBERT)...")

sent_cols = [c for c in df_f.columns if any(x in c for x in ["_pos", "_neg", "_neutral"])]
print(f"🧩 Found sentiment columns: {sent_cols}")

# Gunakan mean agar per hari tidak bias ke jumlah artikel
agg_f = df_f.groupby("date")[sent_cols].mean().reset_index()

# Hitung total mood score (title + content)
agg_f["mood_score"] = (
    agg_f.get("t_pos", 0)
    + agg_f.get("c_pos", 0)
    - agg_f.get("t_neg", 0)
    - agg_f.get("c_neg", 0)
)

# ==========================================================
# MERGE DENGAN MODEL B FINAL
# ==========================================================
print("📊 Merging with ModelB normalized data...")

cols_fx = ["date", "close", "ema20", "ema50", "norm_ema20", "norm_ema50", "norm_close"]
df_b = df_b[[c for c in cols_fx if c in df_b.columns]]

final_df = pd.merge(agg_f, df_b, on="date", how="inner").sort_values("date")

# ==========================================================
# SIMPAN HASIL
# ==========================================================
os.makedirs(os.path.dirname(output_file), exist_ok=True)
final_df.to_excel(output_file, index=False, engine="openpyxl")

print(f"\n✅ Mood series created successfully at:\n{output_file}")
print(f"📏 Final data shape: {final_df.shape}")
print("\n🪄 Preview:")
print(final_df.head(10))
