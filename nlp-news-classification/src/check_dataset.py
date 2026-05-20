import csv
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATASET_URL = "https://www.interpress.com/downloads/interpress_news_category_tr_270k_lite.zip"

ZIP_PATH = DATA_DIR / "interpress_news_category_tr_270k_lite.zip"
EXTRACT_DIR = DATA_DIR / "interpress_news_category_tr_270k_lite"

TRAIN_TSV = EXTRACT_DIR / "interpress_news_category_tr_270k_lite_train.tsv"
TEST_TSV = EXTRACT_DIR / "interpress_news_category_tr_270k_lite_test.tsv"

TRAIN_CSV = DATA_DIR / "train_raw.csv"
TEST_CSV = DATA_DIR / "test_raw.csv"

LABEL_NAMES = [
    "kültürsanat",
    "ekonomi",
    "siyaset",
    "eğitim",
    "dünya",
    "spor",
    "teknoloji",
    "magazin",
    "sağlık",
    "gündem",
]


def increase_csv_field_limit():
    max_size = sys.maxsize

    while True:
        try:
            csv.field_size_limit(max_size)
            print(f"CSV field size limit ayarlandı: {max_size}")
            break
        except OverflowError:
            max_size = int(max_size / 10)


def download_file(url: str, output_path: Path):
    if output_path.exists():
        print(f"Zip dosyası zaten var: {output_path}")
        return

    print(f"Dataset indiriliyor: {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\rİndiriliyor: %{percent:.1f}", end="")

    print()
    print(f"İndirme tamamlandı: {output_path}")


def extract_zip(zip_path: Path, extract_dir: Path):
    if TRAIN_TSV.exists() and TEST_TSV.exists():
        print("Dataset zaten çıkarılmış.")
        return

    print("Zip dosyası açılıyor...")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Çıkarma tamamlandı: {extract_dir}")


def load_tsv(path: Path):
    print(f"Okunuyor: {path}")

    increase_csv_field_limit()

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        engine="python",
    )

    return df


def standardize_dataframe(df: pd.DataFrame):
    print("Orijinal kolonlar:", list(df.columns))

    if "news" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Beklenen kolonlar bulunamadı. Mevcut kolonlar: {list(df.columns)}"
        )

    df = df[["news", "label"]].copy()
    df = df.rename(columns={"news": "content", "label": "category"})

    df["content"] = df["content"].astype(str)
    df["category"] = df["category"].astype(int)
    df["category_name"] = df["category"].apply(lambda x: LABEL_NAMES[x])

    return df


def print_dataset_info(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print()
    print("TRAIN satır sayısı:", len(train_df))
    print("TEST satır sayısı:", len(test_df))

    print()
    print("Kolonlar:")
    print(train_df.columns.tolist())

    print()
    print("İlk train örneği:")
    print(train_df.iloc[0].to_dict())

    print()
    print("Train sınıf dağılımı:")
    print(train_df["category_name"].value_counts().sort_index())

    print()
    print("Test sınıf dağılımı:")
    print(test_df["category_name"].value_counts().sort_index())

    print()
    print("Train metin uzunluğu istatistikleri:")
    train_lengths = train_df["content"].astype(str).str.split().apply(len)
    print(train_lengths.describe())

    print()
    print("Test metin uzunluğu istatistikleri:")
    test_lengths = test_df["content"].astype(str).str.split().apply(len)
    print(test_lengths.describe())

    print()
    print("Eksik değerler - train:")
    print(train_df.isna().sum())

    print()
    print("Eksik değerler - test:")
    print(test_df.isna().sum())


def main():
    download_file(DATASET_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    train_df = load_tsv(TRAIN_TSV)
    test_df = load_tsv(TEST_TSV)

    train_df = standardize_dataframe(train_df)
    test_df = standardize_dataframe(test_df)

    print_dataset_info(train_df, test_df)

    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

    print()
    print(f"Train CSV kaydedildi: {TRAIN_CSV}")
    print(f"Test CSV kaydedildi: {TEST_CSV}")


if __name__ == "__main__":
    main()