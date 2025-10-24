# finetune_longformer_labelledbody_autosave.py
# Fine-tune Longformer on labelled_articles.xlsx (content only)
# with auto-saving predictions every 100 rows to prevent crashing.

import os
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ----------------------
# CONFIGURATION
# ----------------------
INPUT_XLSX = r"D:\Skripsi\Data\Kode Ekstrak\output\labeled_articles.xlsx"
OUTPUT_DIR = r"D:\Skripsi\Data\Kode Ekstrak\output\longformer"
MODEL_NAME = "allenai/longformer-base-4096"

MAX_LENGTH = 1048 # Adjusted max length
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3  # Reduced number of epochs
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLYSTOP_PATIENCE = 80
SEED = 42
TEST_SIZE = 0.1

LABEL_COL = "C_Sentiment"
TEXT_COL = "content"

os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}


# ----------------------
# FUNCTIONS
# ----------------------
def read_dataset(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Input file must contain '{TEXT_COL}' and '{LABEL_COL}' columns")

    df = df[df[TEXT_COL].notna()]
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    df = df.copy().reset_index(drop=True)
    return df


def prepare_datasets(df: pd.DataFrame, tokenizer, test_size=0.1, seed=SEED):
    df["label_text"] = df[LABEL_COL].astype(str).str.lower().str.strip()
    df = df[df["label_text"].isin(label2id.keys())].reset_index(drop=True)
    df["label"] = df["label_text"].map(label2id)

    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])

    ds_train = Dataset.from_pandas(train_df[[TEXT_COL, "label"]])
    ds_eval = Dataset.from_pandas(eval_df[[TEXT_COL, "label"]])

    def tokenize_fn(example):
        return tokenizer(
            example[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    ds_train = ds_train.map(tokenize_fn)
    ds_eval = ds_eval.map(tokenize_fn)

    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    ds_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return ds_train, ds_eval, train_df, eval_df


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": float(acc), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


# ----------------------
# MAIN TRAINING & EVALUATION
# ----------------------
def main():
    print("📥 Loading dataset...")
    df = read_dataset(INPUT_XLSX)
    print(f"Total valid articles: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = 3
    config.id2label = id2label
    config.label2id = label2id

    print("🧠 Loading Longformer model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    ds_train, ds_eval, train_df, eval_df = prepare_datasets(df, tokenizer, test_size=TEST_SIZE)
    print(f"Train size: {len(ds_train)}, Eval size: {len(ds_eval)}")

    training_args = TrainingArguments(
        output_dir=BEST_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        seed=SEED,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLYSTOP_PATIENCE)],
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    trainer.save_model(BEST_MODEL_DIR)

    print("✅ Training complete, generating predictions...")
    preds_output = trainer.predict(ds_eval)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    metrics = compute_metrics((logits, labels))
    print("📊 Evaluation metrics:", metrics)

    # ----------------------
    # AUTO-SAVE PREDICTIONS EVERY 100 ARTICLES
    # ----------------------
    print("💾 Saving predictions every 100 samples...")
    batch_size = 100
    total = len(labels)
    results = []

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        sub_texts = [ds_eval[j][TEXT_COL] for j in range(i, batch_end)]
        sub_true = [id2label[int(labels[j])] for j in range(i, batch_end)]
        sub_pred = [id2label[int(preds[j])] for j in range(i, batch_end)]

        partial_df = pd.DataFrame({
            "content": sub_texts,
            "true_label": sub_true,
            "pred_label": sub_pred,
        })

        # Save partial file
        part_path_csv = os.path.join(OUTPUT_DIR, f"LF_Labelledbody_partial_{i}.csv")
        part_path_xlsx = os.path.join(OUTPUT_DIR, f"LF_Labelledbody_partial_{i}.xlsx")

        partial_df.to_csv(part_path_csv, index=False)
        partial_df.to_excel(part_path_xlsx, index=False, engine="openpyxl")

        print(f"   ↳ Saved {i}–{batch_end} → {os.path.basename(part_path_csv)}")
        results.append(partial_df)

    # Merge all partial results
    df_final = pd.concat(results, ignore_index=True)
    df_final["accuracy"] = metrics["accuracy"]
    df_final["f1"] = metrics["f1"]
    df_final["precision"] = metrics["precision"]
    df_final["recall"] = metrics["recall"]

    final_csv = os.path.join(OUTPUT_DIR, "LF_Labelledbody.csv")
    final_xlsx = os.path.join(OUTPUT_DIR, "LF_Labelledbody.xlsx")

    df_final.to_csv(final_csv, index=False)
    df_final.to_excel(final_xlsx, index=False, engine="openpyxl")

    print(f"🎉 All done. Final results saved to:\n{final_csv}\n{final_xlsx}")
    return metrics


if __name__ == "__main__":
    metrics = main()