import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SEED = 42

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RAW_PATH = DATA_DIR / "train_raw.csv"
TEST_RAW_PATH = DATA_DIR / "test_raw.csv"

TRAIN_PREPARED_PATH = DATA_DIR / "train_prepared.csv"
VAL_PREPARED_PATH = DATA_DIR / "val_prepared.csv"
TEST_PREPARED_PATH = DATA_DIR / "test_prepared.csv"

LABEL_MAP_PATH = DATA_DIR / "label_map.json"
CLASS_DISTRIBUTION_PATH = FIGURE_DIR / "class_distribution_prepared.png"

MAX_TRAIN_PER_CLASS = 4000
MAX_VAL_PER_CLASS = 800
MAX_TEST_PER_CLASS = 1000

MIN_WORDS = 10
MAX_WORDS = 512


def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())
    return text


def add_word_count(df):
    df = df.copy()
    df["word_count"] = df["content"].astype(str).str.split().apply(len)
    return df


def filter_by_length(df):
    df = df.copy()
    df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)]
    return df


def sample_per_class(df, label_col, n_per_class, seed):
    sampled_parts = []

    for label_name, group in df.groupby(label_col):
        n = min(len(group), n_per_class)
        sampled = group.sample(n=n, random_state=seed)
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, axis=0)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    return result


def make_train_val_split(train_df):
    train_parts = []
    val_parts = []

    for label_name, group in train_df.groupby("category_name"):
        group = group.sample(frac=1, random_state=SEED).reset_index(drop=True)

        val_n = min(len(group), MAX_VAL_PER_CLASS)
        train_n = min(len(group) - val_n, MAX_TRAIN_PER_CLASS)

        val_part = group.iloc[:val_n]
        train_part = group.iloc[val_n:val_n + train_n]

        val_parts.append(val_part)
        train_parts.append(train_part)

    train_prepared = pd.concat(train_parts, axis=0)
    val_prepared = pd.concat(val_parts, axis=0)

    train_prepared = train_prepared.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_prepared = val_prepared.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_prepared, val_prepared


def save_class_distribution_plot(train_df, val_df, test_df):
    train_counts = train_df["category_name"].value_counts().sort_index()
    val_counts = val_df["category_name"].value_counts().sort_index()
    test_counts = test_df["category_name"].value_counts().sort_index()

    labels = train_counts.index.tolist()
    x = range(len(labels))

    plt.figure(figsize=(14, 6))

    plt.bar([i - 0.25 for i in x], train_counts.values, width=0.25, label="Train")
    plt.bar(x, val_counts.values, width=0.25, label="Validation")
    plt.bar([i + 0.25 for i in x], test_counts.values, width=0.25, label="Test")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Sample Count")
    plt.title("Prepared Dataset Class Distribution")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(CLASS_DISTRIBUTION_PATH, dpi=200)
    plt.close()

    print(f"Sınıf dağılım grafiği kaydedildi: {CLASS_DISTRIBUTION_PATH}")


def main():
    print("Ham veriler okunuyor...")

    train_raw = pd.read_csv(TRAIN_RAW_PATH)
    test_raw = pd.read_csv(TEST_RAW_PATH)

    print("Ham train:", len(train_raw))
    print("Ham test:", len(test_raw))

    train_raw["content"] = train_raw["content"].apply(clean_text)
    test_raw["content"] = test_raw["content"].apply(clean_text)

    train_raw = add_word_count(train_raw)
    test_raw = add_word_count(test_raw)

    train_raw = filter_by_length(train_raw)
    test_raw = filter_by_length(test_raw)

    print()
    print("Uzunluk filtresinden sonra train:", len(train_raw))
    print("Uzunluk filtresinden sonra test:", len(test_raw))

    train_prepared, val_prepared = make_train_val_split(train_raw)

    test_prepared = sample_per_class(
        test_raw,
        label_col="category_name",
        n_per_class=MAX_TEST_PER_CLASS,
        seed=SEED,
    )

    label_names = sorted(train_prepared["category_name"].unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}

    for df in [train_prepared, val_prepared, test_prepared]:
        df["label"] = df["category_name"].map(label_to_id)

    label_map = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

    train_prepared.to_csv(TRAIN_PREPARED_PATH, index=False, encoding="utf-8-sig")
    val_prepared.to_csv(VAL_PREPARED_PATH, index=False, encoding="utf-8-sig")
    test_prepared.to_csv(TEST_PREPARED_PATH, index=False, encoding="utf-8-sig")

    save_class_distribution_plot(train_prepared, val_prepared, test_prepared)

    print()
    print("Hazırlanan veri setleri:")
    print("Train:", len(train_prepared), TRAIN_PREPARED_PATH)
    print("Validation:", len(val_prepared), VAL_PREPARED_PATH)
    print("Test:", len(test_prepared), TEST_PREPARED_PATH)

    print()
    print("Label map:")
    print(label_to_id)

    print()
    print("Train dağılımı:")
    print(train_prepared["category_name"].value_counts().sort_index())

    print()
    print("Validation dağılımı:")
    print(val_prepared["category_name"].value_counts().sort_index())

    print()
    print("Test dağılımı:")
    print(test_prepared["category_name"].value_counts().sort_index())

    print()
    print("Train metin uzunluğu:")
    print(train_prepared["word_count"].describe())


if __name__ == "__main__":
    main()