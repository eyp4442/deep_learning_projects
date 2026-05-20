import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"

TEST_PATH = DATA_DIR / "test_prepared.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_BINS = 10

TURKISH_STOPWORDS = {
    "ve", "veya", "ile", "de", "da", "ki", "bu", "şu", "o", "bir", "için", "gibi",
    "ama", "fakat", "ancak", "çok", "daha", "en", "son", "olan", "olarak", "ise",
    "diye", "kadar", "sonra", "önce", "her", "hem", "ne", "ya", "mı", "mi", "mu",
    "mü", "nun", "nın", "nin", "ın", "in", "un", "ün", "dan", "den", "tan", "ten",
    "ile", "şekilde", "tarafından", "kendi", "yeni", "göre", "var", "yok", "oldu",
    "olacak", "olduğu", "etti", "ettiği", "dedi", "açıklamada", "belirtti",
}


# -------------------------
# DATASET
# -------------------------

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -------------------------
# UTILS
# -------------------------

def find_latest_berturk_run():
    runs = sorted(
        RUNS_DIR.glob("berturk_news_classifier_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError("BERTurk run klasörü bulunamadı.")

    return runs[0]


def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label_raw = label_map["id_to_label"]

    id_to_label = {
        int(idx): label_name
        for idx, label_name in id_to_label_raw.items()
    }

    class_names = [id_to_label[idx] for idx in sorted(id_to_label.keys())]

    return class_names


def run_inference(model, tokenizer, test_df, device, batch_size):
    dataset = NewsDataset(
        texts=test_df["content"].astype(str).tolist(),
        labels=test_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_true = []
    all_pred = []
    all_probs = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="BERTurk test inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_true), np.array(all_pred), np.array(all_probs)


# -------------------------
# ANALYSIS 1: TOP CONFUSIONS
# -------------------------

def save_top_confused_pairs(y_true, y_pred, class_names, run_dir, top_n=20):
    cm = confusion_matrix(y_true, y_pred)

    pairs = []

    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append({
                    "true_category": class_names[true_idx],
                    "predicted_category": class_names[pred_idx],
                    "count": int(cm[true_idx, pred_idx]),
                })

    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
    pairs_path = run_dir / "top_confused_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")

    top_df = pairs_df.head(top_n).copy()

    labels = [
        f"{row.true_category} → {row.predicted_category}"
        for row in top_df.itertuples()
    ]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_df)), top_df["count"].values)
    plt.yticks(range(len(top_df)), labels, fontsize=9)
    plt.xlabel("Confusion Count")
    plt.title("BERTurk - En Çok Karışan Sınıf Çiftleri")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fig_path = run_dir / "top_confused_pairs.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Top confused pairs kaydedildi: {pairs_path}")
    print(f"Top confused pairs grafiği kaydedildi: {fig_path}")

    return pairs_df


# -------------------------
# ANALYSIS 2: EXAMPLES BY CATEGORY
# -------------------------

def save_correct_wrong_examples(test_df, y_true, y_pred, probs, class_names, run_dir):
    result_df = test_df.copy()

    confidences = probs.max(axis=1)

    result_df["true_label_id"] = y_true
    result_df["pred_label_id"] = y_pred
    result_df["true_category"] = [class_names[i] for i in y_true]
    result_df["predicted_category"] = [class_names[i] for i in y_pred]
    result_df["confidence"] = confidences
    result_df["is_correct"] = result_df["true_label_id"] == result_df["pred_label_id"]
    result_df["short_content"] = result_df["content"].astype(str).str.slice(0, 500)

    examples = []

    for category in class_names:
        category_df = result_df[result_df["true_category"] == category]

        correct = (
            category_df[category_df["is_correct"]]
            .sort_values("confidence", ascending=False)
            .head(3)
        )
        correct = correct.copy()
        correct["example_type"] = "correct_high_confidence"

        wrong = (
            category_df[~category_df["is_correct"]]
            .sort_values("confidence", ascending=False)
            .head(3)
        )
        wrong = wrong.copy()
        wrong["example_type"] = "wrong_high_confidence"

        examples.append(correct)
        examples.append(wrong)

    examples_df = pd.concat(examples, axis=0)

    selected_columns = [
        "example_type",
        "short_content",
        "true_category",
        "predicted_category",
        "confidence",
        "is_correct",
    ]

    examples_path = run_dir / "correct_wrong_examples_by_category.csv"
    examples_df[selected_columns].to_csv(examples_path, index=False, encoding="utf-8-sig")

    full_predictions_path = run_dir / "berturk_full_test_predictions.csv"
    result_df.to_csv(full_predictions_path, index=False, encoding="utf-8-sig")

    print(f"Kategori bazlı doğru/yanlış örnekler kaydedildi: {examples_path}")
    print(f"Tüm test tahminleri kaydedildi: {full_predictions_path}")

    return result_df, examples_df


# -------------------------
# ANALYSIS 3: CALIBRATION / RELIABILITY
# -------------------------

def compute_calibration(y_true, y_pred, probs, run_dir, num_bins=10):
    confidences = probs.max(axis=1)
    correctness = (y_true == y_pred).astype(int)

    bins = np.linspace(0.0, 1.0, num_bins + 1)

    rows = []
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        if i == num_bins - 1:
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(mask.sum())

        if count > 0:
            bin_acc = float(correctness[mask].mean())
            bin_conf = float(confidences[mask].mean())
            bin_fraction = count / len(confidences)
            ece += bin_fraction * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            bin_fraction = 0.0

        rows.append({
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "sample_count": count,
            "accuracy": bin_acc,
            "avg_confidence": bin_conf,
            "fraction": bin_fraction,
        })

    calibration_df = pd.DataFrame(rows)
    calibration_path = run_dir / "calibration_bins.csv"
    calibration_df.to_csv(calibration_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.plot(
        calibration_df["avg_confidence"],
        calibration_df["accuracy"],
        marker="o",
        label="BERTurk",
    )
    plt.xlabel("Average Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram / Calibration Curve\nECE: {ece:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig_path = run_dir / "reliability_diagram.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Calibration bins kaydedildi: {calibration_path}")
    print(f"Reliability diagram kaydedildi: {fig_path}")

    return ece, calibration_df


# -------------------------
# ANALYSIS 4: PRECISION-RECALL
# -------------------------

def save_precision_recall_analysis(y_true, probs, class_names, run_dir):
    y_true_one_hot = np.eye(len(class_names))[y_true]

    ap_rows = []

    plt.figure(figsize=(12, 9))

    for class_idx, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_one_hot[:, class_idx],
            probs[:, class_idx],
        )

        ap = average_precision_score(
            y_true_one_hot[:, class_idx],
            probs[:, class_idx],
        )

        ap_rows.append({
            "category": class_name,
            "average_precision": float(ap),
        })

        plt.plot(recall, precision, label=f"{class_name} AP={ap:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("BERTurk One-vs-Rest Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    pr_path = run_dir / "precision_recall_curves.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    ap_df = pd.DataFrame(ap_rows).sort_values("average_precision", ascending=False)
    ap_path = run_dir / "average_precision_by_class.csv"
    ap_df.to_csv(ap_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.bar(ap_df["category"], ap_df["average_precision"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Precision")
    plt.title("Average Precision by Class")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ap_fig_path = run_dir / "average_precision_by_class.png"
    plt.savefig(ap_fig_path, dpi=200)
    plt.close()

    print(f"Precision-recall grafiği kaydedildi: {pr_path}")
    print(f"Average precision tablosu kaydedildi: {ap_path}")
    print(f"Average precision grafiği kaydedildi: {ap_fig_path}")

    return ap_df


# -------------------------
# ANALYSIS 5: WRONG PREDICTION COMMON WORDS
# -------------------------

def tokenize_for_keyword_analysis(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    tokens = text.split()

    filtered = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in TURKISH_STOPWORDS:
            continue

        filtered.append(token)

    return filtered


def save_wrong_prediction_common_words(result_df, run_dir, top_n=30):
    wrong_df = result_df[~result_df["is_correct"]].copy()

    counter = Counter()

    for text in wrong_df["content"].astype(str):
        counter.update(tokenize_for_keyword_analysis(text))

    most_common = counter.most_common(top_n)

    words_df = pd.DataFrame(most_common, columns=["word", "count"])
    words_path = run_dir / "wrong_prediction_common_words.csv"
    words_df.to_csv(words_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 7))
    plt.barh(range(len(words_df)), words_df["count"].values)
    plt.yticks(range(len(words_df)), words_df["word"].values)
    plt.xlabel("Count")
    plt.title("Yanlış Tahminlerde En Sık Geçen Kelimeler")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fig_path = run_dir / "wrong_prediction_common_words.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Yanlış tahmin ortak kelimeleri kaydedildi: {words_path}")
    print(f"Yanlış tahmin ortak kelime grafiği kaydedildi: {fig_path}")

    return words_df


# -------------------------
# SUMMARY
# -------------------------

def save_summary(run_dir, pairs_df, ece, ap_df, result_df):
    summary_path = run_dir / "detailed_analysis_summary.txt"

    accuracy = result_df["is_correct"].mean()
    wrong_count = int((~result_df["is_correct"]).sum())
    total = len(result_df)

    top_confusions_text = pairs_df.head(10).to_string(index=False)
    ap_text = ap_df.to_string(index=False)

    summary = f"""
BERTurk Detaylı Analiz Özeti
==================================================

Toplam test örneği: {total}
Doğru tahmin oranı: {accuracy:.4f}
Yanlış tahmin sayısı: {wrong_count}

Calibration:
Expected Calibration Error (ECE): {ece:.4f}

En çok karışan ilk 10 sınıf çifti:
{top_confusions_text}

Sınıf bazlı Average Precision:
{ap_text}

Yorum:
- Confusion pair analizi, modelin en çok hangi kategori çiftlerinde zorlandığını gösterir.
- Reliability diagram, model güven skoru ile gerçek doğruluk arasındaki ilişkiyi gösterir.
- Precision-recall analizi, her sınıf için modelin pozitif sınıf ayrım gücünü gösterir.
- Yanlış tahminlerde ortak kelime analizi, hatalı örneklerde hangi konu/kelime alanlarının sık geçtiğini anlamaya yardımcı olur.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Detaylı analiz özeti kaydedildi: {summary_path}")


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="BERTurk run klasörü. Boş bırakılırsa en son BERTurk run kullanılır.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_berturk_run()
    else:
        run_dir = Path(args.run_dir)

    best_model_dir = run_dir / "best_model"

    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model klasörü bulunamadı: {best_model_dir}")

    print("Run directory:", run_dir)
    print("Best model:", best_model_dir)
    print("Batch size:", args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    class_names = load_label_map()

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    test_df = pd.read_csv(TEST_PATH)

    y_true, y_pred, probs = run_inference(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        device=device,
        batch_size=args.batch_size,
    )

    pairs_df = save_top_confused_pairs(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        run_dir=run_dir,
        top_n=20,
    )

    result_df, examples_df = save_correct_wrong_examples(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_names=class_names,
        run_dir=run_dir,
    )

    ece, calibration_df = compute_calibration(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        run_dir=run_dir,
        num_bins=NUM_BINS,
    )

    ap_df = save_precision_recall_analysis(
        y_true=y_true,
        probs=probs,
        class_names=class_names,
        run_dir=run_dir,
    )

    words_df = save_wrong_prediction_common_words(
        result_df=result_df,
        run_dir=run_dir,
        top_n=30,
    )

    save_summary(
        run_dir=run_dir,
        pairs_df=pairs_df,
        ece=ece,
        ap_df=ap_df,
        result_df=result_df,
    )

    print()
    print("Detaylı BERTurk analizi tamamlandı.")
    print(f"Çıktı klasörü: {run_dir}")


if __name__ == "__main__":
    main()