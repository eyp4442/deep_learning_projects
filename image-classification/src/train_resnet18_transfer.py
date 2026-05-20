import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


# -------------------------
# CONFIG
# -------------------------

SEED = 42
NUM_CLASSES = 30
IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_STAGE1 = 5
LR_STAGE1 = 1e-3

EPOCHS_STAGE2 = 8
LR_STAGE2 = 1e-4

WEIGHT_DECAY = 1e-4

TRAIN_DIR = Path("data/stanford_cars/train")
TEST_DIR = Path("data/stanford_cars/test")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES_PATH = OUTPUT_DIR / "selected_classes_30.json"

MODEL_NAME = "resnet18_transfer_30"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = OUTPUT_DIR / "runs" / f"{MODEL_NAME}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / "best_model.pth"

# Eski sabit yolu da koruyoruz.
LEGACY_BEST_MODEL_PATH = MODEL_DIR / "resnet18_transfer_30_best.pth"

REPORT_PATH = RUN_DIR / "classification_report.txt"
CURVE_PATH = RUN_DIR / "curves.png"
HISTORY_PATH = RUN_DIR / "history.json"
METRICS_PATH = RUN_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.png"

AUGMENTATION_EXAMPLES_PATH = RUN_DIR / "augmentation_examples.png"
FILTERS_PATH = RUN_DIR / "first_conv_filters.png"
FEATURE_MAPS_PATH = RUN_DIR / "feature_maps.png"
MODEL_SUMMARY_PATH = RUN_DIR / "model_summary.txt"

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

class StanfordCarsSubsetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def make_filtered_samples(image_folder, selected_classes):
    selected_class_to_new_idx = {
        class_name: new_idx for new_idx, class_name in enumerate(selected_classes)
    }

    idx_to_class = {
        old_idx: class_name for class_name, old_idx in image_folder.class_to_idx.items()
    }

    filtered_samples = []

    for image_path, old_label in image_folder.samples:
        class_name = idx_to_class[old_label]

        if class_name in selected_class_to_new_idx:
            new_label = selected_class_to_new_idx[class_name]
            filtered_samples.append((image_path, new_label))

    return filtered_samples


def load_selected_classes():
    if not SELECTED_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"{SELECTED_CLASSES_PATH} bulunamadı. "
            "Önce baseline veya CarNet dosyalarından biri çalıştırılmış olmalı."
        )

    with open(SELECTED_CLASSES_PATH, "r", encoding="utf-8") as f:
        selected_classes = json.load(f)

    return selected_classes


def prepare_dataloaders():
    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)
    base_test_dataset = datasets.ImageFolder(TEST_DIR)

    selected_classes = load_selected_classes()

    print("Seçilen sınıflar:")
    for class_name in selected_classes:
        print(" -", class_name)

    train_samples_all = make_filtered_samples(base_train_dataset, selected_classes)
    test_samples = make_filtered_samples(base_test_dataset, selected_classes)

    random.shuffle(train_samples_all)

    val_size = int(len(train_samples_all) * 0.2)
    val_samples = train_samples_all[:val_size]
    train_samples = train_samples_all[val_size:]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.04, 0.04),
            scale=(0.92, 1.08),
            shear=3,
        ),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.12,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = StanfordCarsSubsetDataset(train_samples, transform=train_transform)
    val_dataset = StanfordCarsSubsetDataset(val_samples, transform=eval_transform)
    test_dataset = StanfordCarsSubsetDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print()
    print("Train image:", len(train_dataset))
    print("Validation image:", len(val_dataset))
    print("Test image:", len(test_dataset))
    print("Sınıf sayısı:", len(selected_classes))
    print()

    return train_loader, val_loader, test_loader, selected_classes


# -------------------------
# MODEL
# -------------------------

def build_resnet18_transfer(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Başlangıçta tüm pretrained backbone dondurulur.
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def unfreeze_layer4(model):
    # Fine-tuning aşamasında sadece son ResNet bloğu ve classifier açılır.
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


# -------------------------
# TRAIN / EVAL
# -------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, desc):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc=desc, leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating", leave=False)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


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
    plt.plot(epochs, history["val_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Validation Macro-F1 Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"Grafikler kaydedildi: {CURVE_PATH}")


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ResNet18 Transfer Learning - Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Confusion matrix kaydedildi: {output_path}")


def denormalize_image_tensor(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


def save_augmentation_examples(selected_classes, output_path):
    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)

    idx_to_class = {
        old_idx: class_name
        for class_name, old_idx in base_train_dataset.class_to_idx.items()
    }

    selected_class_set = set(selected_classes)

    selected_samples = []

    for image_path, old_label in base_train_dataset.samples:
        class_name = idx_to_class[old_label]

        if class_name in selected_class_set:
            selected_samples.append((image_path, class_name))

    image_path, class_name = random.choice(selected_samples)

    original_image = Image.open(image_path).convert("RGB")
    original_display = original_image.resize((IMG_SIZE, IMG_SIZE))

    augmentation_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.04, 0.04),
            scale=(0.92, 1.08),
            shear=3,
        ),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.12,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
        ),
    ])

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 4, 1)
    plt.imshow(original_display)
    plt.title("Original")
    plt.axis("off")

    for i in range(7):
        augmented_tensor = augmentation_transform(original_image)
        augmented_image = denormalize_image_tensor(augmented_tensor)

        plt.subplot(2, 4, i + 2)
        plt.imshow(augmented_image)
        plt.title(f"Augmented {i + 1}")
        plt.axis("off")

    plt.suptitle(
        f"Data Augmentation Examples\nClass: {class_name}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Augmentation örnekleri kaydedildi: {output_path}")


def normalize_filter_image(filter_tensor):
    filt = filter_tensor.detach().cpu().numpy()
    filt = np.transpose(filt, (1, 2, 0))

    filt_min = filt.min()
    filt_max = filt.max()

    if filt_max - filt_min < 1e-8:
        return np.zeros_like(filt)

    filt = (filt - filt_min) / (filt_max - filt_min)
    return filt


def save_first_conv_filters(model, output_path, max_filters=16):
    conv_layer = model.conv1
    weights = conv_layer.weight.data

    num_filters = min(max_filters, weights.shape[0])

    plt.figure(figsize=(10, 10))

    cols = 4
    rows = int(np.ceil(num_filters / cols))

    for i in range(num_filters):
        filt_img = normalize_filter_image(weights[i])

        plt.subplot(rows, cols, i + 1)
        plt.imshow(filt_img)
        plt.title(f"Filter {i}")
        plt.axis("off")

    plt.suptitle("ResNet18 - First Convolution Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"İlk conv filtreleri kaydedildi: {output_path}")


def save_model_summary(model, output_path):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MODEL SUMMARY - ResNet18 Transfer Learning\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write("=" * 100 + "\n\n")

        f.write("Named Modules:\n")
        f.write("-" * 100 + "\n")

        for name, module in model.named_modules():
            if name == "":
                continue
            f.write(f"{name:50} -> {module.__class__.__name__}\n")

    print(f"Model summary kaydedildi: {output_path}")


def save_feature_maps(model, selected_classes, output_path, device):
    base_test_dataset = datasets.ImageFolder(TEST_DIR)

    idx_to_class = {
        old_idx: class_name
        for class_name, old_idx in base_test_dataset.class_to_idx.items()
    }

    selected_class_set = set(selected_classes)
    selected_samples = []

    for image_path, old_label in base_test_dataset.samples:
        class_name = idx_to_class[old_label]

        if class_name in selected_class_set:
            selected_samples.append((image_path, class_name))

    image_path, class_name = random.choice(selected_samples)

    original_image = Image.open(image_path).convert("RGB")
    original_display = original_image.resize((IMG_SIZE, IMG_SIZE))

    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = eval_tf(original_image).unsqueeze(0).to(device)

    activations = {}

    def get_activation(name):
        def hook(model_module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    hooks = []
    hooks.append(model.conv1.register_forward_hook(get_activation("conv1")))
    hooks.append(model.layer2.register_forward_hook(get_activation("layer2")))
    hooks.append(model.layer4.register_forward_hook(get_activation("layer4")))

    model.eval()

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    plt.figure(figsize=(16, 10))

    # Row 1: Original image
    for i in range(8):
        plt.subplot(4, 8, i + 1)
        if i == 0:
            plt.imshow(original_display)
            plt.title("Original")
        plt.axis("off")

    layer_names = ["conv1", "layer2", "layer4"]

    for row_idx, layer_name in enumerate(layer_names, start=1):
        feature_maps = activations[layer_name][0]
        num_maps = min(8, feature_maps.shape[0])

        for col_idx in range(8):
            plt.subplot(4, 8, row_idx * 8 + col_idx + 1)

            if col_idx < num_maps:
                fmap = feature_maps[col_idx].numpy()
                plt.imshow(fmap, cmap="viridis")

                if col_idx == 0:
                    plt.title(layer_name)

            plt.axis("off")

    plt.suptitle(
        f"ResNet18 Feature Maps\nClass: {class_name}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Feature map görselleri kaydedildi: {output_path}")


def save_history(history):
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.items()
    }

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, indent=4)

    print(f"History kaydedildi: {HISTORY_PATH}")


def save_metrics(metrics):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

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
# TRAINING STAGE
# -------------------------

def run_training_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    selected_classes,
    history,
    best_val_acc,
    stage_name,
    num_epochs,
):
    for epoch in range(num_epochs):
        print(f"{stage_name} - Epoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"{stage_name} Training",
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 70)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "selected_classes": selected_classes,
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "model_name": "ResNet18 Transfer Learning",
                "run_id": RUN_ID,
                "stage_name": stage_name,
            }

            torch.save(checkpoint_data, BEST_MODEL_PATH)

            # Eski sabit path'i de güncel tutuyoruz.
            torch.save(checkpoint_data, LEGACY_BEST_MODEL_PATH)

            print(f"Yeni en iyi ResNet18 modeli kaydedildi: {BEST_MODEL_PATH}")
            print(f"Legacy model yolu güncellendi: {LEGACY_BEST_MODEL_PATH}")
            print()

    return best_val_acc


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
    print("Run ID:", RUN_ID)
    print("Run directory:", RUN_DIR)
    print()

    train_loader, val_loader, test_loader, selected_classes = prepare_dataloaders()

    save_augmentation_examples(
        selected_classes=selected_classes,
        output_path=AUGMENTATION_EXAMPLES_PATH,
    )

    model = build_resnet18_transfer(num_classes=NUM_CLASSES).to(device)

    total_params, trainable_params = count_parameters(model)

    print("MODEL: ResNet18 Transfer Learning")
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre başlangıçta: {trainable_params:,}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_acc = 0.0

    # -------------------------
    # STAGE 1: Feature extraction
    # -------------------------

    print("STAGE 1: Sadece yeni classifier eğitiliyor.")

    optimizer_stage1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1,
        T_max=EPOCHS_STAGE1,
        eta_min=1e-5,
    )

    best_val_acc = run_training_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        device=device,
        selected_classes=selected_classes,
        history=history,
        best_val_acc=best_val_acc,
        stage_name="Stage 1",
        num_epochs=EPOCHS_STAGE1,
    )

    # -------------------------
    # STAGE 2: Fine-tuning
    # -------------------------

    print("STAGE 2: layer4 + classifier fine-tuning başlıyor.")

    unfreeze_layer4(model)

    total_params, trainable_params = count_parameters(model)

    print(f"Eğitilebilir parametre fine-tuning aşamasında: {trainable_params:,}")
    print()

    optimizer_stage2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage2,
        T_max=EPOCHS_STAGE2,
        eta_min=1e-6,
    )

    best_val_acc = run_training_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        device=device,
        selected_classes=selected_classes,
        history=history,
        best_val_acc=best_val_acc,
        stage_name="Stage 2",
        num_epochs=EPOCHS_STAGE2,
    )

    plot_history(history)
    save_history(history)

    print("En iyi ResNet18 modeli test için yükleniyor...")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_model_summary(model, MODEL_SUMMARY_PATH)
    save_first_conv_filters(model, FILTERS_PATH)
    save_feature_maps(
        model=model,
        selected_classes=selected_classes,
        output_path=FEATURE_MAPS_PATH,
        device=device,
    )

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print("RESNET18 TRANSFER LEARNING TEST SONUÇLARI")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test F1:   {test_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=selected_classes,
        zero_division=0,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report kaydedildi: {REPORT_PATH}")

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=selected_classes,
        output_path=CONFUSION_MATRIX_PATH,
    )

    metrics = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "num_classes": NUM_CLASSES,
        "epochs_stage1": EPOCHS_STAGE1,
        "epochs_stage2": EPOCHS_STAGE2,
        "batch_size": BATCH_SIZE,
        "lr_stage1": LR_STAGE1,
        "lr_stage2": LR_STAGE2,
        "weight_decay": WEIGHT_DECAY,
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
        "best_stage": checkpoint.get("stage_name", ""),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "best_model_path": str(BEST_MODEL_PATH),
        "legacy_best_model_path": str(LEGACY_BEST_MODEL_PATH),
        "report_path": str(REPORT_PATH),
        "curve_path": str(CURVE_PATH),
        "history_path": str(HISTORY_PATH),
        "metrics_path": str(METRICS_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "augmentation_examples_path": str(AUGMENTATION_EXAMPLES_PATH),
        "filters_path": str(FILTERS_PATH),
        "feature_maps_path": str(FEATURE_MAPS_PATH),
        "model_summary_path": str(MODEL_SUMMARY_PATH),
    }

    save_metrics(metrics)
    append_metrics_to_csv(metrics)

    print()
    print("Çalıştırma tamamlandı.")
    print(f"Tüm çıktı klasörü: {RUN_DIR}")


if __name__ == "__main__":
    main()