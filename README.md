# Aspect-Based Sentiment Analysis (ABSA)

This project was developed for the DeepX Hackathon.

The goal is to analyze Arabic customer reviews and extract:
1. The aspects mentioned in each review
2. The sentiment for each aspect

The final output is a JSON file following the required submission format.

---

## Task Description

Given a review, the system predicts one or more aspects from:

- food
- service
- price
- cleanliness
- delivery
- ambiance
- app_experience
- general
- none

Then, for each detected aspect, it predicts the sentiment:

- positive
- negative
- neutral

Example:

```json
{
  "review_id": 23,
  "aspects": ["service", "food"],
  "aspect_sentiments": {
    "service": "positive",
    "food": "negative"
  }
}
```

---

## Project Structure

```text
DeepX_Hackathon_neurix/
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│   ├── absa-demo.png
│   └── demo-1.png
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── prepare_data.py
│   ├── train_aspect_model.py
│   ├── train_sentiment_model.py
│   └── predict.py
│
├── outputs/
│   └── predictions.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Approach

The solution is a two-stage pipeline.

### 1. Aspect Detection

A multi-label classifier detects all aspects mentioned in a review.

### 2. Aspect Sentiment Classification

A second model predicts the sentiment for each detected aspect.

Input:

```text
review_text + aspect
```

---

## Model

The models are based on:

```text
UBC-NLP/MARBERTv2
```

MARBERTv2 was selected because the reviews include informal Arabic, mixed language, and real customer expressions.

---

## Data Processing

The preprocessing includes:

- Cleaning text by removing unnecessary symbols and emojis
- Arabic normalization
- Reducing repeated characters
- Keeping meaningful Arabic and English words
- Preparing multi-label targets for aspect detection
- Building a separate dataset for aspect-level sentiment classification

To prepare the data:

```bash
python src/prepare_data.py
```

---

## Training

Train the aspect detection model:

```bash
python src/train_aspect_model.py
```

Train the sentiment classification model:

```bash
python src/train_sentiment_model.py
```

The trained models are saved under:

```text
models/
```

---

## Prediction

Generate the final submission file:

```bash
python src/predict.py
```

The output will be saved as:

```text
outputs/predictions.json
```

---

## Streamlit Demo

Run the local demo application:

```bash
streamlit run app/streamlit_app.py
```

---

## Demo Examples

The system can handle both multi-aspect reviews and ambiguous cases.

### Example 1: Multi-aspect Review

Input:

```text
الأكل كان وحش الخدمة بطيئة والسعر غالي
```

Output:

- food → negative
- service → negative
- price → negative

![Multi Aspect Demo](assets/demo-1.png)

### Example 2: Ambiguous Review

Input:

```text
مش عارف احكم بصراحة
```

Output:

- general → positive

![ABSA Demo](assets/absa-demo.png)

---

## Submission Format

```json
[
  {
    "review_id": 1,
    "aspects": ["service"],
    "aspect_sentiments": {
      "service": "positive"
    }
  }
]
```

Each record contains:

- `review_id`
- `aspects`
- `aspect_sentiments`

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---
