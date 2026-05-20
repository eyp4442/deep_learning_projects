import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# -------------------------
# CONFIG
# -------------------------

MODEL_NAME = "tfidf_logistic_regression"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RUN_DIR = OUTPUT_DIR / "runs" / f"{MODEL_NAME}_{RUN_ID}"

RUN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_prepared.csv"
VAL_PATH = DATA_DIR / "val_prepared.csv"
TEST_PATH = DATA_DIR / "test_prepared.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

MODEL_PATH = RUN_DIR / "tfidf_logreg_model.joblib"
VECTORIZER_PATH = RUN_DIR / "tfidf_vectorizer.joblib"
REPORT_PATH = RUN_DIR / "classification_report.txt"
METRICS_PATH = RUN_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.png"
EXPERIMENTS_CSV_PATH = OUTPUT_DIR / "experiment_results.csv"

MAX_FEATURES = 100_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95


# -------------------------
# UTILS
# -------------------------

def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label_raw = label_map["id_to_label"]

    # JSON key'leri string olarak okuyabilir; int'e çeviriyoruz.
    id_to_label = {
        int(idx): label_name
        for idx, label_name in id_to_label_raw.items()
    }

    class_names = [
        id_to_label[idx]
        for idx in sorted(id_to_label.keys())
    ]

    return class_names


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("TF-IDF + Logistic Regression Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    print(f"Confusion matrix kaydedildi: {CONFUSION_MATRIX_PATH}")


def save_metrics(metrics):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Metrikler kaydedildi: {METRICS_PATH}")


def append_metrics_to_csv(metrics):
    file_exists = EXPERIMENTS_CSV_PATH.exists()

    with open(EXPERIMENTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)

    print(f"Deney sonucu CSV dosyasına eklendi: {EXPERIMENTS_CSV_PATH}")


# -------------------------
# MAIN
# -------------------------

def main():
    print("Model:", MODEL_NAME)
    print("Run ID:", RUN_ID)
    print("Run directory:", RUN_DIR)
    print()

    class_names = load_label_map()
    train_df, val_df, test_df = load_data()

    X_train = train_df["content"].astype(str).tolist()
    y_train = train_df["label"].astype(int).values

    X_val = val_df["content"].astype(str).tolist()
    y_val = val_df["label"].astype(int).values

    X_test = test_df["content"].astype(str).tolist()
    y_test = test_df["label"].astype(int).values

    print()
    print("TF-IDF vectorizer oluşturuluyor...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Validation TF-IDF shape:", X_val_tfidf.shape)
    print("Test TF-IDF shape:", X_test_tfidf.shape)

    print()
    print("Logistic Regression eğitiliyor...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        verbose=0,
    )

    model.fit(X_train_tfidf, y_train)

    print()
    print("Validation değerlendiriliyor...")

    val_pred = model.predict(X_val_tfidf)

    val_acc = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

    print(f"Validation Accuracy:    {val_acc:.4f}")
    print(f"Validation Macro-F1:    {val_macro_f1:.4f}")
    print(f"Validation Weighted-F1: {val_weighted_f1:.4f}")

    print()
    print("Test değerlendiriliyor...")

    test_pred = model.predict(X_test_tfidf)

    test_acc = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test Macro-F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted-F1: {test_weighted_f1:.4f}")

    report = classification_report(
        y_test,
        test_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report kaydedildi: {REPORT_PATH}")

    save_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        class_names=class_names,
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Vectorizer kaydedildi: {VECTORIZER_PATH}")

    metrics = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "num_classes": len(class_names),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_features": MAX_FEATURES,
        "ngram_range": str(NGRAM_RANGE),
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_weighted_f1": float(val_weighted_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "report_path": str(REPORT_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
    }

    save_metrics(metrics)
    append_metrics_to_csv(metrics)

    print()
    print("TF-IDF baseline tamamlandı.")
    print(f"Tüm çıktı klasörü: {RUN_DIR}")


if __name__ == "__main__":
    main()