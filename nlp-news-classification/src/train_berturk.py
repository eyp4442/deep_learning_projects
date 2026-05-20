import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# -------------------------
# CONFIG
# -------------------------

SEED = 42

MODEL_NAME = "berturk_news_classifier"
HF_MODEL_NAME = "dbmdz/bert-base-turkish-cased"

NUM_CLASSES = 10
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TRAIN_PATH = DATA_DIR / "train_prepared.csv"
VAL_PATH = DATA_DIR / "val_prepared.csv"
TEST_PATH = DATA_DIR / "test_prepared.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = OUTPUT_DIR / "runs" / f"{MODEL_NAME}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_DIR = RUN_DIR / "best_model"
REPORT_PATH = RUN_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.png"
CURVE_PATH = RUN_DIR / "curves.png"
HISTORY_PATH = RUN_DIR / "history.json"
METRICS_PATH = RUN_DIR / "metrics.json"

EXPERIMENTS_CSV_PATH = OUTPUT_DIR / "experiment_results.csv"


# -------------------------
# REPRODUCIBILITY
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        return item


def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label_raw = label_map["id_to_label"]

    id_to_label = {
        int(idx): label_name
        for idx, label_name in id_to_label_raw.items()
    }

    class_names = [
        id_to_label[idx]
        for idx in sorted(id_to_label.keys())
    ]

    return class_names, id_to_label


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    return train_df, val_df, test_df


def create_dataloaders(tokenizer):
    train_df, val_df, test_df = load_data()

    train_dataset = NewsDataset(
        texts=train_df["content"].astype(str).tolist(),
        labels=train_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    val_dataset = NewsDataset(
        texts=val_df["content"].astype(str).tolist(),
        labels=val_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    test_dataset = NewsDataset(
        texts=test_df["content"].astype(str).tolist(),
        labels=test_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader, train_df, val_df, test_df


# -------------------------
# TRAIN / EVAL
# -------------------------

def compute_class_weights(train_df, device):
    y_train = train_df["label"].astype(int).values

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("Class weights:")
    for idx, weight in enumerate(class_weights.detach().cpu().numpy()):
        print(f"  Class {idx}: {weight:.4f}")

    return class_weights


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating", leave=False)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item() * input_ids.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return (
        epoch_loss,
        epoch_acc,
        epoch_macro_f1,
        epoch_weighted_f1,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# -------------------------
# VISUALIZATION / SAVING
# -------------------------

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(16, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Macro-F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_macro_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_macro_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"Grafikler kaydedildi: {CURVE_PATH}")


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("BERTurk News Classifier Confusion Matrix")
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


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    print(f"History kaydedildi: {HISTORY_PATH}")


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


def save_prediction_examples(test_df, y_true, y_pred, probs, class_names):
    examples_path = RUN_DIR / "prediction_examples.csv"

    confidences = probs.max(axis=1)

    result_df = test_df.copy()
    result_df["true_label_id"] = y_true
    result_df["pred_label_id"] = y_pred
    result_df["true_label_name"] = [class_names[i] for i in y_true]
    result_df["pred_label_name"] = [class_names[i] for i in y_pred]
    result_df["confidence"] = confidences
    result_df["is_correct"] = result_df["true_label_id"] == result_df["pred_label_id"]

    # Rapor için metni çok uzun bırakmayalım.
    result_df["short_content"] = result_df["content"].astype(str).str.slice(0, 350)

    selected_columns = [
        "short_content",
        "true_label_name",
        "pred_label_name",
        "confidence",
        "is_correct",
    ]

    examples = []

    # Yüksek güvenli doğru örnekler
    correct_high = result_df[result_df["is_correct"]].sort_values(
        "confidence",
        ascending=False,
    ).head(10)
    correct_high["example_type"] = "correct_high_confidence"
    examples.append(correct_high[selected_columns + ["example_type"]])

    # Yüksek güvenli yanlış örnekler
    wrong_high = result_df[~result_df["is_correct"]].sort_values(
        "confidence",
        ascending=False,
    ).head(10)
    wrong_high["example_type"] = "wrong_high_confidence"
    examples.append(wrong_high[selected_columns + ["example_type"]])

    # Düşük güvenli örnekler
    low_confidence = result_df.sort_values(
        "confidence",
        ascending=True,
    ).head(10)
    low_confidence["example_type"] = "low_confidence"
    examples.append(low_confidence[selected_columns + ["example_type"]])

    examples_df = pd.concat(examples, axis=0)
    examples_df.to_csv(examples_path, index=False, encoding="utf-8-sig")

    print(f"Tahmin örnekleri kaydedildi: {examples_path}")


# -------------------------
# MAIN
# -------------------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Model name:", MODEL_NAME)
    print("HF model:", HF_MODEL_NAME)
    print("Run ID:", RUN_ID)
    print("Run directory:", RUN_DIR)
    print()

    class_names, id_to_label = load_label_map()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_dataloaders(
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)},
    )

    model.to(device)

    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print()
    print("Training configuration:")
    print("MAX_LENGTH:", MAX_LENGTH)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("Total steps:", total_steps)
    print("Warmup steps:", warmup_steps)
    print()

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }

    best_val_macro_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        (
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            _,
            _,
            _,
        ) = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_macro_f1"].append(float(train_macro_f1))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["val_weighted_f1"].append(float(val_weighted_f1))

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro-F1: {train_macro_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Macro-F1:   {val_macro_f1:.4f} | Val Weighted-F1: {val_weighted_f1:.4f}")
        print("-" * 90)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1

            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

            print(f"Yeni en iyi BERTurk modeli kaydedildi: {BEST_MODEL_DIR}")
            print()

    plot_history(history)
    save_history(history)

    print("En iyi model test için yükleniyor...")

    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.to(device)

    (
        test_loss,
        test_acc,
        test_macro_f1,
        test_weighted_f1,
        y_true,
        y_pred,
        probs,
    ) = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("BERTURK TEST SONUÇLARI")
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test Macro-F1:    {test_macro_f1:.4f}")
    print(f"Test Weighted-F1: {test_weighted_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report kaydedildi: {REPORT_PATH}")

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    save_prediction_examples(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        class_names=class_names,
    )

    metrics = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "hf_model_name": HF_MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "best_model_dir": str(BEST_MODEL_DIR),
        "report_path": str(REPORT_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "curve_path": str(CURVE_PATH),
        "history_path": str(HISTORY_PATH),
    }

    save_metrics(metrics)
    append_metrics_to_csv(metrics)

    print()
    print("BERTurk eğitimi tamamlandı.")
    print(f"Tüm çıktı klasörü: {RUN_DIR}")


if __name__ == "__main__":
    main()