import html
import re
import pandas as pd
from src.constants import (
    EMAIL_RE,
    TFIDF_MAX_FEATURES,
    RANDOM_STATE,
    TFIDF_NGRAM_RANGE,
    TEST_SIZE,
    TFIDF_MIN_DF,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def clean_text(s: str) -> str:
    if len(s) == 0:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)  # replace URLs
    s = re.sub(r"[^A-Za-z0-9@._\-'\s]", " ", s)  # keep common chars
    s = re.sub(r"\s+", " ", s).strip().lower()  # What does this do?
    return s


def extract_sender_email_domain(sender):
    if pd.isna(sender):
        return "", ""
    s = str(sender)
    m = EMAIL_RE.search(s)
    if m:
        return m.group(0).lower(), m.group(2).lower()
    return "", ""


def preprocess_emails(data: pd.DataFrame):
    # ---- Preprocess text data to remove non next data ----
    print("[INFO] Cleaning text fields...")
    data["subject_clean"] = data["subject"].apply(clean_text)
    data["body_clean"] = data["body"].apply(clean_text)
    data["text"] = (
            data["subject_clean"].fillna("") + " " + data["body_clean"].fillna("")
    ).str.strip()


    data["subject_len"] = data["subject_clean"].str.len().fillna(0)
    data["body_len"] = data["body_clean"].str.len().fillna(0)
    data["exclaim_count"] = data["body"].fillna("").astype(str).str.count("!")

    y = data["label"]
    X = data[["text", "subject_len", "body_len", "exclaim_count"]]

    print("[INFO] Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    num_cols = [
        "subject_len",
        "body_len",
        "exclaim_count",
    ]

    col_transformer = ColumnTransformer(transformers=[
        (make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
            TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,  # (1,1) = unigrams only
            min_df=TFIDF_MIN_DF,)
        ),['text']),
        (make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler()
        ),num_cols)
    ])

    X_train = col_transformer.fit_transform(X_train)
    X_test = col_transformer.transform(X_test)
    # TF-IDF on combined text

    # ---- Train/test split ----
    return X_train, X_test, y_train, y_test, col_transformer  # coerce label and urls
