import ast
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train_fixed.xlsx"
VALIDATION_PATH = PROJECT_ROOT / "data" / "raw" / "validation_fixed.xlsx"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ASPECTS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none",
]

ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = ARABIC_DIACRITICS.sub("", text)

    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", " ", text)

    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    return ast.literal_eval(value)


def parse_dict(value):
    if isinstance(value, dict):
        return value

    if pd.isna(value):
        return {}

    return ast.literal_eval(value)


def validate_labels(df, name):
    invalid_rows = []

    for idx, row in df.iterrows():
        aspects = set(row["aspects"])
        sentiment_keys = set(row["aspect_sentiments"].keys())

        if aspects != sentiment_keys:
            invalid_rows.append(idx)

    if invalid_rows:
        raise ValueError(
            f"{name} has invalid label rows. "
            f"Aspects and sentiment keys do not match at rows: {invalid_rows[:10]}"
        )

    print(f"{name} labels are valid.")


def add_aspect_columns(df):
    for aspect in ASPECTS:
        df[f"aspect_{aspect}"] = df["aspects"].apply(
            lambda aspects: 1 if aspect in aspects else 0
        )

    return df


def build_sentiment_dataset(df):
    rows = []

    for _, row in df.iterrows():
        for aspect in row["aspects"]:
            rows.append(
                {
                    "review_id": row["review_id"],
                    "text": row["clean_text"],
                    "aspect": aspect,
                    "input_text": f"{row['clean_text']} [ASP] {aspect}",
                    "sentiment": row["aspect_sentiments"][aspect],
                }
            )

    return pd.DataFrame(rows)


def prepare_dataframe(path, name):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_excel(path)

    required_columns = ["review_id", "review_text", "aspects", "aspect_sentiments"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{name} is missing columns: {missing_columns}")

    df["aspects"] = df["aspects"].apply(parse_list)
    df["aspect_sentiments"] = df["aspect_sentiments"].apply(parse_dict)
    df["clean_text"] = df["review_text"].apply(clean_text)

    validate_labels(df, name)

    df = add_aspect_columns(df)

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = prepare_dataframe(TRAIN_PATH, "train")
    validation_df = prepare_dataframe(VALIDATION_PATH, "validation")

    train_sentiment_df = build_sentiment_dataset(train_df)
    validation_sentiment_df = build_sentiment_dataset(validation_df)

    train_df.to_pickle(PROCESSED_DIR / "train_processed.pkl")
    validation_df.to_pickle(PROCESSED_DIR / "validation_processed.pkl")

    train_sentiment_df.to_pickle(PROCESSED_DIR / "train_sentiment.pkl")
    validation_sentiment_df.to_pickle(PROCESSED_DIR / "validation_sentiment.pkl")

    print("Data preparation completed successfully.")
    print("Saved files:")
    print(PROCESSED_DIR / "train_processed.pkl")
    print(PROCESSED_DIR / "validation_processed.pkl")
    print(PROCESSED_DIR / "train_sentiment.pkl")
    print(PROCESSED_DIR / "validation_sentiment.pkl")

    print("\nTrain shape:", train_df.shape)
    print("Validation shape:", validation_df.shape)
    print("Train sentiment shape:", train_sentiment_df.shape)
    print("Validation sentiment shape:", validation_sentiment_df.shape)


if __name__ == "__main__":
    main()