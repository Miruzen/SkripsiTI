import os
import pandas as pd

# 📂 Lokasi folder tempat file partial disimpan
BASE_DIR = r"D:\Skripsi\Data\Kode Ekstrak\output\model\LongFormer"

# 🔍 Ambil semua file partial
csv_files = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.startswith("LF_Labelledbody_partial_") and f.endswith(".csv")
])

xlsx_files = sorted([
    os.path.join(BASE_DIR, f)
    for f in os.listdir(BASE_DIR)
    if f.startswith("LF_Labelledbody_partial_") and f.endswith(".xlsx")
])

print(f"📁 CSV ditemukan: {len(csv_files)} file")
print(f"📄 XLSX ditemukan: {len(xlsx_files)} file")

# 🧩 Validasi kecocokan jumlah file
if len(csv_files) != len(xlsx_files):
    print("⚠️ PERINGATAN: Jumlah file CSV dan XLSX tidak sama!")
    missing_csv = [os.path.basename(f).replace('.xlsx', '.csv') for f in xlsx_files if os.path.basename(f).replace('.xlsx', '.csv') not in [os.path.basename(x) for x in csv_files]]
    missing_xlsx = [os.path.basename(f).replace('.csv', '.xlsx') for f in csv_files if os.path.basename(f).replace('.csv', '.xlsx') not in [os.path.basename(x) for x in xlsx_files]]
    if missing_csv:
        print(f"❌ File CSV hilang untuk: {missing_csv}")
    if missing_xlsx:
        print(f"❌ File XLSX hilang untuk: {missing_xlsx}")
else:
    print("✅ Jumlah file CSV dan XLSX cocok.")

# 🧠 Gabungkan semua CSV dan XLSX
print("\n🔄 Menggabungkan file CSV...")
merged_csv = pd.concat((pd.read_csv(f, encoding='utf-8') for f in csv_files), ignore_index=True)

print("🔄 Menggabungkan file XLSX...")
merged_xlsx = pd.concat((pd.read_excel(f) for f in xlsx_files), ignore_index=True)

# 🧹 Hapus duplikasi (berdasarkan isi teks & label prediksi)
merged_csv.drop_duplicates(inplace=True)
merged_xlsx.drop_duplicates(inplace=True)

# 📊 Informasi hasil akhir
print(f"\n📈 Total baris hasil gabungan CSV : {len(merged_csv)}")
print(f"📈 Total baris hasil gabungan XLSX: {len(merged_xlsx)}")

# 💾 Simpan hasil final
final_csv = os.path.join(BASE_DIR, "LF_Labelled.csv")
final_xlsx = os.path.join(BASE_DIR, "LF_Labelled.xlsx")

merged_csv.to_csv(final_csv, index=False, encoding='utf-8-sig')
merged_xlsx.to_excel(final_xlsx, index=False, engine="openpyxl")

print("\n✅ Semua file berhasil digabungkan dan disimpan!")
print(f"📄 CSV: {final_csv}")
print(f"📊 XLSX: {final_xlsx}")
