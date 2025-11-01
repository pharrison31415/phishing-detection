import re
import csv
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def set_csv_field_size_limit():
    """Increase CSV field size limit for very long lines."""
    max_int = csv.field_size_limit()
    try:
        csv.field_size_limit(2**31 - 1)
    except OverflowError:
        # Fallback just in case
        csv.field_size_limit(max_int)


# =========================
# Main
# =========================
def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    set_csv_field_size_limit()

    # ---- Load ----
    print(f"[INFO] Loading csv")
    df = pd.read_csv(
        "data/CEAS_08.csv",
        sep=",",
        quotechar='"',
        engine="python",  # robust to quoted newlines
        on_bad_lines="warn",
    )

    expected_cols = {"sender", "receiver", "date", "subject", "body", "label", "urls"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected column(s): {missing}. Found: {list(df.columns)}"
        )

    # ---- Preprocess ----
    print("[INFO] Cleaning text fields...")
    df["subject_clean"] = df["subject"].apply(clean_text)
    df["body_clean"] = df["body"].apply(clean_text)
    df["text"] = (
        df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")
    ).str.strip()

    # ---- Metadata segmentation ----
    print("[INFO] Extracting metadata features...")
    df[["sender_email", "sender_domain"]] = df["sender"].apply(
        lambda s: pd.Series(extract_sender_email_domain(s))
    )

    df["url_count_body"] = df["body"].apply(count_urls)
    df["url_count_subj"] = df["subject"].apply(count_urls)
    df["subject_len"] = df["subject_clean"].str.len().fillna(0)
    df["body_len"] = df["body_clean"].str.len().fillna(0)
    df["exclaim_count"] = df["body"].fillna("").astype(str).str.count("!")

    for kw in KEYWORDS:
        df[f"kw_{kw}"] = (
            (df["subject_clean"].str.contains(rf"\b{re.escape(kw)}\b", na=False))
            | (df["body_clean"].str.contains(rf"\b{re.escape(kw)}\b", na=False))
        ).astype(int)

    # coerce label and urls
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["urls_numeric"] = (
        pd.to_numeric(df["urls"], errors="coerce").fillna(0).astype(int)
    )

    # ---- Feature representation ----
    print("[INFO] Building features (TF-IDF + domain + numeric metadata)...")

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

    # Combine sparse blocks
    X = hstack([X_text, X_domain, X_num_sparse]).tocsr()

    # ---- Train/test split ----
    print("[INFO] Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ---- Models ----
    models = {
        "Dummy (Majority)": DummyClassifier(strategy="most_frequent"),
        "RandomForest": RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "LinearSVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }

    # ---- Train & evaluate ----
    print("[INFO] Training and evaluating models...")
    records = []
    reports = {}
    cms = {}

    for name, model in models.items():
        print(f"[INFO] -> {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        records.append(
            {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        )
        reports[name] = classification_report(y_test, y_pred, digits=3, zero_division=0)
        cms[name] = confusion_matrix(y_test, y_pred)

    results_df = (
        pd.DataFrame(records).sort_values("F1", ascending=False).reset_index(drop=True)
    )

    # ---- Save artifacts ----
    metrics_path = ARTIFACT_DIR / "CEAS08_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    report_paths = {}
    for name, rep in reports.items():
        p = ARTIFACT_DIR / f"report_{name.replace(' ', '_')}.txt"
        with open(p, "w") as f:
            f.write(rep)
        report_paths[name] = str(p)

    cm_paths = {}
    for name, cm in cms.items():
        p = ARTIFACT_DIR / f"confusion_{name.replace(' ', '_')}.csv"
        pd.DataFrame(
            cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
        ).to_csv(p)
        cm_paths[name] = str(p)

    # ---- Print summary ----
    print("\n=== Model Evaluation Summary (higher is better) ===")
    print(results_df.to_string(index=False))
    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    for name, p in report_paths.items():
        print(f"[INFO] {name} report: {p}")
    for name, p in cm_paths.items():
        print(f"[INFO] {name} confusion matrix: {p}")


if __name__ == "__main__":
    main()
