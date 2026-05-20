import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"
TEST_PATH = Path("data/test_prepared.csv")

DISPLAY_MAX_LENGTH = 80


def find_latest_berturk_run():
    runs = sorted(
        RUNS_DIR.glob("berturk_news_classifier_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError("BERTurk run klasörü bulunamadı.")

    return runs[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="BERTurk run klasörü. Boş bırakılırsa en son BERTurk run kullanılır.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=5,
        help="Kaç örneğin tokenization çıktısı üretilecek.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_berturk_run()
    else:
        run_dir = Path(args.run_dir)

    best_model_dir = run_dir / "best_model"

    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model klasörü bulunamadı: {best_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)

    test_df = pd.read_csv(TEST_PATH)
    sample_df = test_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    output_txt_path = run_dir / "tokenization_examples.txt"
    output_csv_path = run_dir / "tokenization_examples.csv"

    rows = []

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("BERTurk Tokenization Examples\n")
        f.write("=" * 80 + "\n\n")

        for idx, row in sample_df.iterrows():
            text = str(row["content"])
            category = str(row["category_name"])

            encoded = tokenizer(
                text,
                add_special_tokens=True,
                max_length=DISPLAY_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            f.write(f"Örnek {idx + 1}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Gerçek kategori: {category}\n")
            f.write(f"Orijinal metin kısa hali:\n{text[:500]}\n\n")

            f.write("Tokenlar:\n")
            f.write(" ".join(tokens) + "\n\n")

            f.write("Input IDs:\n")
            f.write(str(input_ids) + "\n\n")

            f.write("Attention Mask:\n")
            f.write(str(attention_mask) + "\n\n")

            f.write("=" * 80 + "\n\n")

            rows.append({
                "category": category,
                "short_content": text[:500],
                "tokens": " ".join(tokens),
                "input_ids": " ".join(str(x) for x in input_ids),
                "attention_mask": " ".join(str(x) for x in attention_mask),
            })

    pd.DataFrame(rows).to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"Tokenization TXT kaydedildi: {output_txt_path}")
    print(f"Tokenization CSV kaydedildi: {output_csv_path}")

    print()
    print("İlk örnek tokenları:")
    print(rows[0]["tokens"])


if __name__ == "__main__":
    main()