#!/usr/bin/env python3
"""
Input  : data/combined_articles_cleaned.xlsx
Output : output/artikel/labeled_articles_final.xlsx
Columns: Date | title | T_Predict | T_Pos | T_Neg | T_Neutral
"""

import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
INPUT_FILE = "/content/drive/MyDrive/Skripsi/data/combined_articles_cleaned.xlsx"
OUTPUT_FILE = "/content/drive/MyDrive/Skripsi/output/labeled_articles_final.xlsx"
MODEL_NAME = "ProsusAI/finbert"

os.makedirs("data", exist_ok=True)
os.makedirs("output/artikel", exist_ok=True)

# -----------------------------------------------------------
# Initialize Model
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

labels = ["negative", "neutral", "positive"]

# -----------------------------------------------------------
# Helper Function
# -----------------------------------------------------------
def analyze_sentiment(text: str):
    """Analyze sentiment for a given text (title only)."""
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0, 0.0, 1.0  # default neutral

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    probs = probs.cpu().numpy()
    neg, neu, pos = probs.tolist()

    label_id = torch.argmax(outputs.logits, dim=-1).item()
    label = labels[label_id]

    return label, round(pos, 4), round(neg, 4), round(neu, 4)

# -----------------------------------------------------------
# Load and Process Data
# -----------------------------------------------------------
print(f"📂 Loading dataset from {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

required_cols = ["date", "title"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column '{col}' in input file")

results = []

print("🔍 Analyzing title sentiment using FinBERT...")
try:
    # Moved tqdm outside the loop to create a single progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing titles"):
        title = str(row["title"])
        date = str(row["date"])

        t_label, t_pos, t_neg, t_neu = analyze_sentiment(title)

        results.append({
            "Date": date,
            "Title": title,
            "T_Predict": t_label,
            "T_Pos": t_pos,
            "T_Neg": t_neg,
            "T_Neutral": t_neu
        })
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # -----------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
        print(f"✅ Progress saved to {OUTPUT_FILE}")
    else:
        print("No results to save.")

print(f"✅ Title sentiment labeling complete! Saved to {OUTPUT_FILE}")