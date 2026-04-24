import json
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


ASPECTS = [
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience", "general", "none"
]

SENTIMENTS = ["positive", "negative", "neutral"]

TEST_PATH = "data/raw/unlabeled_fixed.xlsx"
ASPECT_MODEL_DIR = "models/aspect_model"
SENTIMENT_MODEL_DIR = "models/sentiment_model"
OUTPUT_PATH = "outputs/predictions.json"


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", " ", text)

    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_df = pd.read_excel(TEST_PATH)
    test_df["clean_text"] = test_df["review_text"].apply(clean_text)

    aspect_tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_DIR)
    aspect_model = AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL_DIR).to(device)
    aspect_model.eval()

    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR).to(device)
    sentiment_model.eval()

    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
        review_id = int(row["review_id"])
        text = row["clean_text"]
    
        # Aspect prediction
        encoded = aspect_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=160,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = aspect_model(**encoded).logits.cpu().numpy()[0]

        probs = sigmoid(logits)

        predicted_aspects = [
            aspect for aspect, prob in zip(ASPECTS, probs)
            if prob >= 0.4
        ]

        if len(predicted_aspects) == 0:
            predicted_aspects = ["general"]

        if "none" in predicted_aspects and len(predicted_aspects) > 1:
            predicted_aspects.remove("none")

        aspect_sentiments = {}

        for aspect in predicted_aspects:
            sentiment_input = text + " [ASP] " + aspect

            sentiment_encoded = sentiment_tokenizer(
                sentiment_input,
                truncation=True,
                padding="max_length",
                max_length=192,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                sentiment_logits = sentiment_model(**sentiment_encoded).logits.cpu().numpy()[0]

            sentiment_id = int(np.argmax(sentiment_logits))
            aspect_sentiments[aspect] = SENTIMENTS[sentiment_id]

        predictions.append({
            "review_id": review_id,
            "aspects": predicted_aspects,
            "aspect_sentiments": aspect_sentiments
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {OUTPUT_PATH}")
    print("Total predictions:", len(predictions))


if __name__ == "__main__":
    main()