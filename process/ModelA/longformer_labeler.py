#!/usr/bin/env python3
"""
Input  : /data/combined_articles_cleaned.xlsx
Output : /output/artikel/labeled_articles_final.xlsx
Columns: date | title | content | C_Predict | C_Pos | C_Neutral | C_Neg
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
INPUT_FILE = "/content/drive/MyDrive/Skripsi/data/combined_articles_cleaned.xlsx"
OUTPUT_FILE = "/content/drive/MyDrive/Skripsi/output/labeled_articles_final2.xlsx"
MODEL_PATH = "/content/drive/MyDrive/Skripsi/output/LongFormer/best_model"  
SAVE_EVERY = 500 
BATCH_SIZE = 4 
TEMP_SAVE_DIR = os.path.dirname(OUTPUT_FILE)
TEMP_SAVE_PREFIX = os.path.basename(OUTPUT_FILE).replace(".xlsx", "_partial")

os.makedirs(TEMP_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
print("🧠 Loading Longformer model from local path...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

id2label = {0: "negative", 1: "neutral", 2: "positive"}

# -----------------------------------------------------------
# SENTIMENT ANALYSIS FUNCTION
# -----------------------------------------------------------
def analyze_batch(texts, tokenizer, model, device, max_len=4096):
    """Analyze sentiment for a batch of texts."""
    if not texts:
        return [], [], [], []

    processed_texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]

    inputs = tokenizer(
        processed_texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predictions = torch.argmax(probs, dim=-1).tolist()
    probs_list = probs.tolist()

    labels = [id2label[pred] for pred in predictions]
    pos_probs = [round(p[2], 4) for p in probs_list]
    neu_probs = [round(p[1], 4) for p in probs_list]
    neg_probs = [round(p[0], 4) for p in probs_list]

    return labels, pos_probs, neu_probs, neg_probs

# -----------------------------------------------------------
# LOAD DATA AND CHECK FOR PARTIAL SAVE
# -----------------------------------------------------------
print(f"📂 Loading dataset from {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

if "content" not in df.columns:
    raise ValueError("❌ Missing 'content' column in input file")

# Check for existing partial save files
partial_files = sorted([f for f in os.listdir(TEMP_SAVE_DIR) if f.startswith(TEMP_SAVE_PREFIX) and f.endswith(".xlsx")])

start_index = 0
if partial_files:
    latest_partial_file = partial_files[-1]
    latest_partial_path = os.path.join(TEMP_SAVE_DIR, latest_partial_file)
    print(f"✅ Found partial save file: {latest_partial_file}. Resuming from where it left off.")
    df_partial = pd.read_excel(latest_partial_path)
    start_index = len(df_partial)
    df.loc[df_partial.index, ["C_Predict", "C_Pos", "C_Neutral", "C_Neg"]] = df_partial[["C_Predict", "C_Pos", "C_Neutral", "C_Neg"]]


if start_index == 0:
    df["C_Predict"] = ""
    df["C_Pos"] = 0.0
    df["C_Neutral"] = 0.0
    df["C_Neg"] = 0.0


# -----------------------------------------------------------
# PROCESS ARTICLES IN BATCHES
# -----------------------------------------------------------
print("🔍 Starting sentiment inference for article bodies...")
try:
    for i in tqdm(range(start_index, len(df), BATCH_SIZE), desc="Processing articles in batches"):
        batch_texts = df["content"][i : i + BATCH_SIZE].tolist()
        labels, pos_probs, neu_probs, neg_probs = analyze_batch(
            batch_texts, tokenizer, model, device
        )

        df.loc[i : i + BATCH_SIZE - 1, "C_Predict"] = labels
        df.loc[i : i + BATCH_SIZE - 1, "C_Pos"] = pos_probs
        df.loc[i : i + BATCH_SIZE - 1, "C_Neutral"] = neu_probs
        df.loc[i : i + BATCH_SIZE - 1, "C_Neg"] = neg_probs

        if (i + BATCH_SIZE) % SAVE_EVERY == 0:
            processed_count = i + BATCH_SIZE
            temp_path = os.path.join(TEMP_SAVE_DIR, f"{TEMP_SAVE_PREFIX}_{processed_count}.xlsx")
            df.iloc[:processed_count].to_excel(temp_path, index=False, engine="openpyxl")
            print(f"\n💾 Auto-saved progress → {temp_path}") 


except Exception as e:
    print(f"\n🚨 An error occurred: {e}")
    current_processed_count = i + len(labels)
    temp_path = os.path.join(TEMP_SAVE_DIR, f"{TEMP_SAVE_PREFIX}_crash_at_{current_processed_count}.xlsx")
    df.iloc[:current_processed_count].to_excel(temp_path, index=False, engine="openpyxl")
    print(f"💾 Progress saved to {temp_path} due to error.")


# -----------------------------------------------------------
# SAVE FINAL RESULT AND CLEANUP PARTIAL FILES
# -----------------------------------------------------------
if start_index + len(df[start_index:]) == len(df):
    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print(f"✅ Sentiment labeling complete! Saved to {OUTPUT_FILE}")
    for partial_file in partial_files:
        os.remove(os.path.join(TEMP_SAVE_DIR, partial_file))
    print("🧹 Cleaned up partial save files.")
else:
    print(f"❌ Process interrupted. Partial results saved incrementally.")