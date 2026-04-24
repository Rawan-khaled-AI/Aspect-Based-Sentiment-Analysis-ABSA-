import re
import numpy as np
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

SENTIMENTS = ["positive", "negative", "neutral"]

ASPECT_MODEL_DIR = "models/aspect_model"
SENTIMENT_MODEL_DIR = "models/sentiment_model"


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
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


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aspect_tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_DIR)
    aspect_model = AutoModelForSequenceClassification.from_pretrained(
        ASPECT_MODEL_DIR
    ).to(device)
    aspect_model.eval()

    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_DIR
    ).to(device)
    sentiment_model.eval()

    return device, aspect_tokenizer, aspect_model, sentiment_tokenizer, sentiment_model


def predict_review(text):
    threshold = 0.4  # ثابت

    device, aspect_tokenizer, aspect_model, sentiment_tokenizer, sentiment_model = load_models()

    clean = clean_text(text)

    encoded = aspect_tokenizer(
        clean,
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
        if prob >= threshold
    ]

    if not predicted_aspects:
        predicted_aspects = ["general"]

    if "none" in predicted_aspects and len(predicted_aspects) > 1:
        predicted_aspects.remove("none")

    aspect_sentiments = {}

    for aspect in predicted_aspects:
        sentiment_input = clean + " [ASP] " + aspect

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

    return clean, predicted_aspects, aspect_sentiments


# ---------------- UI ----------------

st.set_page_config(
    page_title="Arabic ABSA",
    page_icon="💬",
    layout="centered",
)

st.title("Arabic Aspect-Based Sentiment Analysis")

st.write(
    "Enter an Arabic customer review and the model will detect aspects and sentiment for each aspect."
)

review_text = st.text_area(
    "Review text",
    height=160,
    placeholder="مثال: الاكل كان حلو جدا بس الخدمة بطيئة والسعر غالي",
)

if st.button("Analyze"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        clean, aspects, sentiments = predict_review(review_text)

        st.subheader("Cleaned Text")
        st.write(clean)

        st.subheader("Predicted Aspects")
        st.write(aspects)

        st.subheader("Aspect Sentiments")

        for aspect in aspects:
            sentiment = sentiments[aspect]

            if sentiment == "positive":
                st.success(f"{aspect}: {sentiment}")
            elif sentiment == "negative":
                st.error(f"{aspect}: {sentiment}")
            else:
                st.info(f"{aspect}: {sentiment}")

        st.subheader("JSON Output")
        st.json({
            "aspects": aspects,
            "aspect_sentiments": sentiments,
        })