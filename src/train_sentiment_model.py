import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

MODEL_NAME = "UBC-NLP/MARBERTv2"

SENTIMENTS = ["positive", "negative", "neutral"]
SENTIMENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENTIMENT = {i: s for s, i in SENTIMENT2ID.items()}

TRAIN_PATH = "data/processed/train_sentiment.pkl"
VAL_PATH = "data/processed/validation_sentiment.pkl"
OUTPUT_DIR = "models/sentiment_model"


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=192):
        self.texts = df["input_text"].tolist()
        self.labels = df["sentiment"].map(SENTIMENT2ID).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    print("GPU available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    train_df = pd.read_pickle(TRAIN_PATH)
    val_df = pd.read_pickle(VAL_PATH)

    print(train_df["sentiment"].value_counts())
    print(val_df["sentiment"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(SENTIMENTS),
        id2label=ID2SENTIMENT,
        label2id=SENTIMENT2ID,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Sentiment model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()