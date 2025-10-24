#!/usr/bin/env python3
"""
finbert_labeler.py
Analyze sentiment of combined forex articles using FinBERT (ProsusAI/finbert).

Input  : /data/combined_articles.xlsx
Output : /output/labeled_articles.xlsx
Columns: Date | title | T_Sentiment | T_Score | Content | C_Sentiment | C_Score
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
INPUT_FILE = "data/combined_articles_cleaned.xlsx"
OUTPUT_FILE = "output/artikel/labeled_articles_final.xlsx"
MODEL_NAME = "ProsusAI/finbert"

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

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
# Helper: Chunked Sentiment Analysis
# -----------------------------------------------------------
def analyze_sentiment(text: str, tokenizer, model, device):
    """Analyze sentiment for arbitrarily long text by chunking."""
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.0

    tokens = tokenizer.tokenize(text)
    chunk_size = 510  # max 512 - CLS/SEP
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    all_probs = []

    for chunk in chunks:
        inputs = tokenizer.encode_plus(
            chunk,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    label_id = torch.argmax(avg_probs).item()
    label = labels[label_id]
    score = avg_probs[label_id].item()
    return label, round(score, 4)

# -----------------------------------------------------------
# Load and Process Data
# -----------------------------------------------------------
print(f"📂 Loading dataset from {INPUT_FILE} ...")
df = pd.read_excel(INPUT_FILE)

required_cols = ["date", "title", "content"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column '{col}' in input file")

results = []

print("🔍 Analyzing sentiment... This may take a while.")
for _, row in tqdm(df.iterrows(), total=len(df)):
    title = str(row["title"])
    content = str(row["content"])
    date = str(row["date"])

    # Analyze both title and content
    t_label, t_score = analyze_sentiment(title, tokenizer, model, device)
    c_label, c_score = analyze_sentiment(content, tokenizer, model, device)

    results.append({
        "Date": date,
        "title": title,
        "T_Sentiment": t_label,
        "T_Score": t_score,
        "Content": content,
        "C_Sentiment": c_label,
        "C_Score": c_score
    })

# -----------------------------------------------------------
# Save Results
# -----------------------------------------------------------
df_out = pd.DataFrame(results)
df_out.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"✅ Sentiment labeling complete! Saved to {OUTPUT_FILE}")
