import csv
import json
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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
IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_STAGE1 = 5
LR_STAGE1 = 8e-4

EPOCHS_STAGE2 = 8
LR_STAGE2 = 1e-4

EPOCHS_STAGE3 = 6
LR_STAGE3 = 3e-5

WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.20

TRAIN_DIR = Path("data/stanford_cars/train")
TEST_DIR = Path("data/stanford_cars/test")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "resnet18_full196_v2"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = OUTPUT_DIR / "runs" / f"{MODEL_NAME}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / "best_model.pth"

# Eski sabit yolu da koruyoruz.
LEGACY_BEST_MODEL_PATH = MODEL_DIR / "resnet18_full196_v2_best.pth"

REPORT_PATH = RUN_DIR / "classification_report.txt"
CURVE_PATH = RUN_DIR / "curves.png"
HISTORY_PATH = RUN_DIR / "history.json"
METRICS_PATH = RUN_DIR / "metrics.json"

TOP_CONFUSIONS_PATH = RUN_DIR / "top_confusions.png"
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

class StanfordCarsDataset(Dataset):
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


def stratified_train_val_split(samples, val_ratio=0.2):
    class_to_samples = defaultdict(list)

    for path, label in samples:
        class_to_samples[label].append((path, label))

    train_samples = []
    val_samples = []

    for label, class_samples in class_to_samples.items():
        random.shuffle(class_samples)

        val_size = max(1, int(len(class_samples) * val_ratio))

        val_samples.extend(class_samples[:val_size])
        train_samples.extend(class_samples[val_size:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def get_train_transform():
    return transforms.Compose([
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


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def prepare_dataloaders():
    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)
    base_test_dataset = datasets.ImageFolder(TEST_DIR)

    selected_classes = base_train_dataset.classes
    num_classes = len(selected_classes)

    print("Full Stanford Cars sınıf sayısı:", num_classes)
    print("İlk 10 sınıf:")
    for class_name in selected_classes[:10]:
        print(" -", class_name)

    train_samples_all = base_train_dataset.samples
    test_samples = base_test_dataset.samples

    train_samples, val_samples = stratified_train_val_split(
        train_samples_all,
        val_ratio=VAL_RATIO,
    )

    train_dataset = StanfordCarsDataset(train_samples, transform=get_train_transform())
    val_dataset = StanfordCarsDataset(val_samples, transform=get_eval_transform())
    test_dataset = StanfordCarsDataset(test_samples, transform=get_eval_transform())

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
    print("Sınıf sayısı:", num_classes)
    print()

    return train_loader, val_loader, test_loader, selected_classes


# -------------------------
# MODEL
# -------------------------

def build_resnet18_transfer(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Başlangıçta bütün pretrained backbone dondurulur.
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def unfreeze_layer4(model):
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer3_and_layer4(model):
    for param in model.layer3.parameters():
        param.requires_grad = True

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

def compute_top5_correct(outputs, labels):
    _, top5_preds = outputs.topk(5, dim=1)
    correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
    return correct.any(dim=1).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device, desc):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    top5_correct = 0
    total_samples = 0

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

        top5_correct += compute_top5_correct(outputs.detach(), labels.detach())
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_top5 = top5_correct / total_samples

    return epoch_loss, epoch_acc, epoch_top5


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    top5_correct = 0
    total_samples = 0

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

            top5_correct += compute_top5_correct(outputs.detach(), labels.detach())
            total_samples += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_top5 = top5_correct / total_samples

    return epoch_loss, epoch_acc, epoch_f1, epoch_top5, all_labels, all_preds


# -------------------------
# VISUALIZATION / SAVING
# -------------------------

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(18, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    # Macro-F1
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["val_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Validation Macro-F1 Curve")
    plt.legend()
    plt.grid(True)

    # Top-5
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_top5"], label="Train Top-5")
    plt.plot(epochs, history["val_top5"], label="Validation Top-5")
    plt.xlabel("Epoch")
    plt.ylabel("Top-5 Accuracy")
    plt.title("Top-5 Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    print(f"Grafikler kaydedildi: {CURVE_PATH}")


def denormalize_image_tensor(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


def save_augmentation_examples(output_path):
    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)

    image_path, label = random.choice(base_train_dataset.samples)
    class_name = base_train_dataset.classes[label]

    original_image = Image.open(image_path).convert("RGB")
    original_display = original_image.resize((IMG_SIZE, IMG_SIZE))

    augmentation_transform = get_train_transform()

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

    plt.suptitle("ResNet18 Full196 v2 - First Convolution Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"İlk conv filtreleri kaydedildi: {output_path}")


def save_model_summary(model, output_path):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MODEL SUMMARY - ResNet18 Full196 v2\n")
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


def save_feature_maps(model, class_names, output_path, device):
    base_test_dataset = datasets.ImageFolder(TEST_DIR)

    image_path, label = random.choice(base_test_dataset.samples)
    class_name = class_names[label]

    original_image = Image.open(image_path).convert("RGB")
    original_display = original_image.resize((IMG_SIZE, IMG_SIZE))

    input_tensor = get_eval_transform()(original_image).unsqueeze(0).to(device)

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
        f"ResNet18 Full196 v2 Feature Maps\nClass: {class_name}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Feature map görselleri kaydedildi: {output_path}")


def save_top_confusions(y_true, y_pred, class_names, output_path, top_n=20):
    cm = confusion_matrix(y_true, y_pred)

    pairs = []

    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append((
                    cm[true_idx, pred_idx],
                    class_names[true_idx],
                    class_names[pred_idx],
                ))

    pairs.sort(reverse=True, key=lambda x: x[0])
    top_pairs = pairs[:top_n]

    labels = [f"{true_class}\n→ {pred_class}" for count, true_class, pred_class in top_pairs]
    values = [count for count, true_class, pred_class in top_pairs]

    plt.figure(figsize=(14, 8))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), labels, fontsize=7)
    plt.xlabel("Confusion Count")
    plt.title("ResNet18 Full196 v2 - Most Confused Class Pairs")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"En çok karıştırılan sınıflar grafiği kaydedildi: {output_path}")


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

        train_loss, train_acc, train_top5 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"{stage_name} Training",
        )

        val_loss, val_acc, val_f1, val_top5, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_top5"].append(float(train_top5))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))
        history["val_top5"].append(float(val_top5))

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top-5: {train_top5:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | Val Top-5: {val_top5:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 90)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "selected_classes": selected_classes,
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "val_top5": float(val_top5),
                "model_name": "ResNet18 Full196 v2",
                "run_id": RUN_ID,
                "stage_name": stage_name,
            }

            torch.save(checkpoint_data, BEST_MODEL_PATH)

            # Eski sabit path'i de güncel tutuyoruz.
            torch.save(checkpoint_data, LEGACY_BEST_MODEL_PATH)

            print(f"Yeni en iyi ResNet18 Full196 v2 modeli kaydedildi: {BEST_MODEL_PATH}")
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

    save_augmentation_examples(output_path=AUGMENTATION_EXAMPLES_PATH)

    num_classes = len(selected_classes)
    model = build_resnet18_transfer(num_classes=num_classes).to(device)

    total_params, trainable_params = count_parameters(model)

    print("MODEL: ResNet18 Transfer Learning Full 196 v2")
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre başlangıçta: {trainable_params:,}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_top5": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_top5": [],
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
    # STAGE 2: layer4 fine-tuning
    # -------------------------

    print("STAGE 2: layer4 + classifier fine-tuning başlıyor.")

    unfreeze_layer4(model)

    total_params, trainable_params = count_parameters(model)

    print(f"Eğitilebilir parametre Stage 2 aşamasında: {trainable_params:,}")
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

    # -------------------------
    # STAGE 3: layer3 + layer4 fine-tuning
    # -------------------------

    print("STAGE 3: layer3 + layer4 + classifier fine-tuning başlıyor.")

    unfreeze_layer3_and_layer4(model)

    total_params, trainable_params = count_parameters(model)

    print(f"Eğitilebilir parametre Stage 3 aşamasında: {trainable_params:,}")
    print()

    optimizer_stage3 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE3,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_stage3 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage3,
        T_max=EPOCHS_STAGE3,
        eta_min=1e-6,
    )

    best_val_acc = run_training_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage3,
        scheduler=scheduler_stage3,
        device=device,
        selected_classes=selected_classes,
        history=history,
        best_val_acc=best_val_acc,
        stage_name="Stage 3",
        num_epochs=EPOCHS_STAGE3,
    )

    plot_history(history)
    save_history(history)

    print("En iyi ResNet18 Full196 v2 modeli test için yükleniyor...")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    save_model_summary(model, MODEL_SUMMARY_PATH)
    save_first_conv_filters(model, FILTERS_PATH)
    save_feature_maps(
        model=model,
        class_names=selected_classes,
        output_path=FEATURE_MAPS_PATH,
        device=device,
    )

    test_loss, test_acc, test_f1, test_top5, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print("RESNET18 FULL 196 v2 TRANSFER LEARNING TEST SONUÇLARI")
    print(f"Test Loss:  {test_loss:.4f}")
    print(f"Test Acc:   {test_acc:.4f}")
    print(f"Test F1:    {test_f1:.4f}")
    print(f"Test Top-5: {test_top5:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=selected_classes,
        zero_division=0,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Classification report kaydedildi: {REPORT_PATH}")

    save_top_confusions(
        y_true=y_true,
        y_pred=y_pred,
        class_names=selected_classes,
        output_path=TOP_CONFUSIONS_PATH,
        top_n=20,
    )

    metrics = {
        "model_name": MODEL_NAME,
        "run_id": RUN_ID,
        "num_classes": num_classes,
        "epochs_stage1": EPOCHS_STAGE1,
        "epochs_stage2": EPOCHS_STAGE2,
        "epochs_stage3": EPOCHS_STAGE3,
        "batch_size": BATCH_SIZE,
        "lr_stage1": LR_STAGE1,
        "lr_stage2": LR_STAGE2,
        "lr_stage3": LR_STAGE3,
        "weight_decay": WEIGHT_DECAY,
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
        "best_val_top5": float(checkpoint["val_top5"]),
        "best_stage": checkpoint.get("stage_name", ""),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "test_top5": float(test_top5),
        "best_model_path": str(BEST_MODEL_PATH),
        "legacy_best_model_path": str(LEGACY_BEST_MODEL_PATH),
        "report_path": str(REPORT_PATH),
        "curve_path": str(CURVE_PATH),
        "history_path": str(HISTORY_PATH),
        "metrics_path": str(METRICS_PATH),
        "top_confusions_path": str(TOP_CONFUSIONS_PATH),
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