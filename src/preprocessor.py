import html
import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.constants import (
    EMAIL_RE,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE
)
from sklearn.model_selection import train_test_split


def clean_text(s: str) -> str:
    if type(s) != str:
        return ""

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
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    # ---- Train/test split ----
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    num_cols = [
        "subject_len",
        "body_len",
        "exclaim_count",
    ]
    # Ensure a 1D sequence of strings is passed to TfidfVectorizer
    text_selector = FunctionTransformer(lambda df: df["text"], validate=False)

    col_transformer = ColumnTransformer(transformers=[
        ("text", make_pipeline(
            text_selector,
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,  # (1,1) = unigrams only
                min_df=TFIDF_MIN_DF,
                max_df=TFIDF_MAX_DF,
            )

        ), ['text']),
        ("numeric", make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler()
        ), num_cols)
    ])

    X_train = col_transformer.fit_transform(X_train)
    X_test = col_transformer.transform(X_test)
    return X_train, X_test, col_transformer
