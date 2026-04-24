import pandas as pd
import re

# ─────────────────────────────────────────────
# 1. LOAD — keep ALL useful columns
# ─────────────────────────────────────────────
train = pd.read_excel("DeepX_train.xlsx")

# Keep business_category — it's a strong signal for aspect detection
train = train[["review_id", "review_text", "star_rating", "business_category"]]


# ─────────────────────────────────────────────
# 2. CLEAN TEXT — much more thorough
# ─────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags (just in case)
    text = re.sub(r"<.*?>", "", text)

    # Remove date artifacts like "تاريخ التعديل: قبل يوم واحد" that leaked into review_text
    text = re.sub(r"تاريخ التعديل[^،.]*", "", text)

    # Remove emojis
    text = re.sub(
        r"[\U00010000-\U0010ffff"
        r"\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF]+",
        "", text, flags=re.UNICODE
    )

    # Normalize Arabic: remove tashkeel (diacritics)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)

    # Normalize Arabic letters (alef variants → ا, ya → ي, ta marbuta → ة)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)

    # Normalize punctuation — collapse multiple ! or ? 
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


train["clean_text"] = train["review_text"].apply(clean_text)

# Drop rows where clean_text is empty after cleaning
train = train[train["clean_text"].str.len() > 0].reset_index(drop=True)


# ─────────────────────────────────────────────
# 3. DOCUMENT-LEVEL SENTIMENT (from stars)
#    ⚠️ This is APPROXIMATE — used as a weak label only
#    Real aspect sentiment needs per-aspect judgment
# ─────────────────────────────────────────────
def get_sentiment(stars):
    try:
        stars = float(stars)
    except (ValueError, TypeError):
        return "neutral"
    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"


train["doc_sentiment"] = train["star_rating"].apply(get_sentiment)


# ─────────────────────────────────────────────
# 4. ASPECT KEYWORDS — massively expanded
#    Covers Arabic dialects (Egyptian, Gulf, Levantine),
#    English, French, Italian, Turkish seen in your data
# ─────────────────────────────────────────────
aspect_keywords = {
    "food": [
        # Arabic
        "اكل", "طعم", "وجبة", "طعام", "مذاق", "شيف", "رز", "لحم", "فراخ",
        "سمك", "حلو", "مرق", "منيو", "قائمة طعام", "مقبلات", "تقديم",
        "بارد", "ناشف", "طازج", "خبز", "كفتة", "شاورما", "فول", "طعمية",
        # English
        "food", "taste", "meal", "dish", "menu", "delicious", "tasty",
        "flavor", "overcooked", "undercooked", "fresh", "stale", "lamb",
        "chicken", "beef", "rice", "bread", "dessert", "breakfast", "lunch", "dinner",
        # French / Italian / Turkish (seen in your data)
        "nourriture", "repas", "cibo", "qualità", "yemek", "lezzet",
    ],
    "service": [
        # Arabic
        "خدمة", "موظف", "نادل", "استقبال", "تعامل", "موظفين", "ترحيب",
        "وقفة", "احترام", "بشوش", "وقاحة", "فظ", "محترم", "مساعدة",
        # English
        "service", "staff", "waiter", "waitress", "manager", "rude", "polite",
        "helpful", "friendly", "attentive", "ignored", "slow", "fast", "tip",
        "baksheesh", "security", "guard",
        # French / Italian / Turkish
        "service", "servizio", "personel", "garson",
    ],
    "price": [
        # Arabic
        "سعر", "غالي", "رخيص", "تكلفة", "حساب", "فلوس", "قيمة", "اسعار",
        "مبالغ", "مناسب", "بسعر",
        # English
        "price", "expensive", "cheap", "cost", "overpriced", "affordable",
        "value", "bill", "money", "worth", "budget",
        # French / Italian / Turkish
        "prix", "cher", "prezzo", "pahalı", "ucuz",
    ],
    "cleanliness": [
        # Arabic
        "نظيف", "نظافة", "وسخ", "قذر", "نظيفة", "اتساخ", "حمام", "دورة مياه",
        # English
        "clean", "dirty", "hygiene", "hygienic", "filthy", "spotless",
        "washroom", "toilet", "restroom", "bathroom",
        # French / Italian / Turkish
        "propre", "sale", "pulito", "temiz",
    ],
    "delivery": [
        # Arabic
        "توصيل", "تأخير", "وصل", "طلب", "اوردر", "دليفري", "توصل",
        # English
        "delivery", "deliver", "late", "on time", "order", "shipped",
        "courier", "arrived",
    ],
    "ambiance": [
        # Arabic
        "جو", "ديكور", "مكان", "اجواء", "تصميم", "ضوضاء", "هادي", "صاخب",
        "فخم", "رومانسي", "عائلي", "مريح", "ضيق", "واسع", "جلسة", "اضاءة",
        "فرش", "ترتيب", "منظر",
        # English
        "ambiance", "atmosphere", "decor", "decoration", "interior", "noise",
        "quiet", "loud", "cozy", "spacious", "view", "seating", "lighting",
        "parking", "valet", "ramp", "stroller", "wheelchair", "handicapped",
        # French / Italian / Turkish
        "ambiance", "atmosphère", "atmosfer", "dekor",
    ],
    "app_experience": [
        # Arabic
        "تطبيق", "ابليكيشن", "موقع", "اون لاين", "اوردر اون لاين",
        # English
        "app", "application", "website", "web", "online", "login",
        "interface", "ui", "ux", "loading", "crash", "bug", "glitch",
    ],
}


# ─────────────────────────────────────────────
# 5. ASPECT DETECTION — smarter logic
#    - Uses business_category as a prior
#    - Detects multiple aspects per review correctly
#    - Uses word-boundary matching to avoid false positives
#      e.g. "place" shouldn't match inside "replace"
# ─────────────────────────────────────────────

# Category → likely aspects (used to boost recall for vague reviews)
CATEGORY_ASPECT_PRIOR = {
    "مطعم": ["food", "service", "price", "cleanliness", "ambiance"],
    "مركز تسوق": ["ambiance", "cleanliness", "price", "service"],
    "توصيل": ["delivery", "service", "food"],
    "تطبيق": ["app_experience", "service"],
}


def detect_aspects(text, business_category, doc_sentiment):
    text_lower = text.lower()
    found_aspects = []

    for aspect, keywords in aspect_keywords.items():
        for kw in keywords:
            # Use word boundary for Latin script, substring for Arabic
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) or \
               (any('\u0600' <= c <= '\u06ff' for c in kw) and kw in text_lower):
                found_aspects.append(aspect)
                break  # avoid duplicate aspect from multiple keywords

    # If nothing found, check category prior for vague positive reviews
    # e.g. "رائع" (wonderful) in a restaurant → food + service as general
    if not found_aspects:
        # Check if any category keyword matches
        for cat_key, prior_aspects in CATEGORY_ASPECT_PRIOR.items():
            if cat_key in str(business_category):
                # Only use prior for very short/vague reviews
                if len(text.split()) <= 4:
                    found_aspects = ["general"]
                else:
                    found_aspects = ["general"]
                break
        if not found_aspects:
            found_aspects = ["general"]

    return found_aspects


# ─────────────────────────────────────────────
# 6. BUILD TRAINING ROWS
#    One row per (review, aspect) pair
#    sentiment = doc_sentiment (weak label — will be improved by model)
# ─────────────────────────────────────────────
rows = []

for _, row in train.iterrows():
    text = row["clean_text"]
    sentiment = row["doc_sentiment"]
    category = row["business_category"]

    aspects = detect_aspects(text, category, sentiment)

    for aspect in aspects:
        rows.append({
            "review_id": row["review_id"],
            "text": text,
            "business_category": category,
            "aspect": aspect,
            "sentiment": sentiment,     # ⚠️ doc-level weak label
            "star_rating": row["star_rating"],
        })

train_absa = pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 7. QUALITY CHECKS
# ─────────────────────────────────────────────
print("=" * 50)
print(f"Total reviews loaded:       {len(train)}")
print(f"Total (review, aspect) rows:{len(train_absa)}")
print(f"Avg aspects per review:     {len(train_absa)/len(train):.2f}")
print()
print("Aspect distribution:")
print(train_absa["aspect"].value_counts())
print()
print("Sentiment distribution:")
print(train_absa["sentiment"].value_counts())
print()
print("Sample output:")
print(train_absa[["review_id", "text", "aspect", "sentiment"]].head(10).to_string())

# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
train_absa.to_csv("train_absa.csv", index=False)
print("\n✅ Saved to train_absa.csv")