import html
import re
import pandas as pd
from src.constants import (
    EMAIL_RE,
    URL_RE,
    KEYWORDS,
    TFIDF_MAX_FEATURES,
    RANDOM_STATE,
    TFIDF_NGRAM_RANGE,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler


def clean_text(s: str) -> str:
    if len(s) == 0:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)  # replace URLs
    s = re.sub(r"[^A-Za-z0-9@._\-'\s]", " ", s)  # keep common chars
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def extract_sender_email_domain(sender):
    if pd.isna(sender):
        return "", ""
    s = str(sender)
    m = EMAIL_RE.search(s)
    if m:
        return m.group(0).lower(), m.group(2).lower()
    return "", ""


# TODO: Remove this as the data already includes url counts + clean text makes them countable the CountVectorizer
def count_urls(s):
    if pd.isna(s):
        return 0
    return len(URL_RE.findall(str(s)))


def preprocess_emails(data: pd.DataFrame):
    # ---- Preprocess text data to remove non next data ----
    print("[INFO] Cleaning text fields...")
    data["subject_clean"] = data["subject"].apply(clean_text)
    data["body_clean"] = data["body"].apply(clean_text)
    data["text"] = (
        data["subject_clean"].fillna("") + " " + data["body_clean"].fillna("")
    ).str.strip()

    # ---- Metadata segmentation ----
    print("[INFO] Extracting metadata features...")
    data[["sender_email", "sender_domain"]] = data["sender"].apply(
        lambda s: pd.Series(extract_sender_email_domain(s))
    )

    data["url_count_body"] = data["body"].apply(count_urls)
    data["url_count_subj"] = data["subject"].apply(count_urls)
    data["subject_len"] = data["subject_clean"].str.len().fillna(0)
    data["body_len"] = data["body_clean"].str.len().fillna(0)
    data["exclaim_count"] = data["body"].fillna("").astype(str).str.count("!")

    for kw in KEYWORDS:
        data[f"kw_{kw}"] = (
            (data["subject_clean"].str.contains(rf"\b{re.escape(kw)}\b", na=False))
            | (data["body_clean"].str.contains(rf"\b{re.escape(kw)}\b", na=False))
        ).astype(int)

    # coerce label and urls
    # TODO: Why are we coercing these? We should use an imputer
    y = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
    data["urls_numeric"] = (
        pd.to_numeric(data["urls"], errors="coerce").fillna(0).astype(int)
    )

    # TF-IDF on combined text
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,  # (1,1) = unigrams only
        min_df=TFIDF_MIN_DF,
    )
    X_text = tfidf.fit_transform(df["text"].fillna(""))

    # Domains (CountVectorizer, min_df=2 to drop ultra-rare)
    domain_vectorizer = CountVectorizer(min_df=2)
    X_domain = domain_vectorizer.fit_transform(df["sender_domain"].fillna(""))

    # Numeric features
    num_cols = [
        "url_count_body",
        "url_count_subj",
        "subject_len",
        "body_len",
        "exclaim_count",
        "urls_numeric",
    ] + [f"kw_{kw}" for kw in KEYWORDS]

    X_num = df[num_cols].fillna(0).astype(float).values

    # Standardize numeric features (with_mean=False for sparse compatibility)
    scaler = StandardScaler(with_mean=False)
    X_num_sparse = csr_matrix(scaler.fit_transform(X_num))

    # Combine sparse blocks # TODO: Why is this necessary?
    X = hstack([X_text, X_domain, X_num_sparse]).tocsr()

    # ---- Train/test split ----
    # TODO: Train/test split should happen before scaling
    print("[INFO] Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test
