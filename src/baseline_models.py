import csv
import pandas as pd
from src.preprocessor import preprocess_emails, scale_features
from src.constants import (
    ARTIFACT_DIR,
    RANDOM_STATE,
    RF_TREES
)
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
    """Increase CSV field size limit for very long lines,"""
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
    print("[INFO] Loading csv")
    cols = ["subject", "body", "label"]
    df = pd.read_csv("data/Enron.csv", usecols=cols)
    # Use concat to stack datasets (merge performs joins and can drop rows)
    df = pd.concat(
        [
            df,
            pd.read_csv("data/Nazario.csv", usecols=cols),
            pd.read_csv("data/Nigerian_Fraud.csv", usecols=cols),
        ],
        ignore_index=True,
    )

    expected_cols = {"subject", "body", "label"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected column(s): {missing}. Found: {list(df.columns)}"
        )

    # ---- Preprocess ----
    X_train, X_test, y_train, y_test = preprocess_emails(df)

    X_train, X_test, col_transformer = scale_features(X_train, X_test)
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
    metrics_path = ARTIFACT_DIR / "performance_metrics.csv"
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