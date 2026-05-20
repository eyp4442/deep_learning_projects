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
from tqdm import tqdm


# -------------------------
# CONFIG
# -------------------------

SEED = 42
NUM_CLASSES = 30
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 2e-4

TRAIN_DIR = Path("data/stanford_cars/train")
TEST_DIR = Path("data/stanford_cars/test")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES_PATH = OUTPUT_DIR / "selected_classes_30.json"

MODEL_NAME = "carnet_v2"
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = OUTPUT_DIR / "runs" / f"{MODEL_NAME}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RUN_DIR / "best_model.pth"

# Eski sabit yolu da koruyoruz.
# Başka scriptler carnet_v2_30_best.pth dosyasını ararsa bozulmaz.
LEGACY_BEST_MODEL_PATH = MODEL_DIR / "carnet_v2_30_best.pth"

REPORT_PATH = RUN_DIR / "classification_report.txt"
CURVE_PATH = RUN_DIR / "curves.png"
HISTORY_PATH = RUN_DIR / "history.json"
METRICS_PATH = RUN_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = RUN_DIR / "confusion_matrix.png"
AUGMENTATION_EXAMPLES_PATH = RUN_DIR / "augmentation_examples.png"
MODEL_SUMMARY_PATH = RUN_DIR / "model_summary.txt"
FIRST_CONV_FILTERS_PATH = RUN_DIR / "first_conv_filters.png"
FEATURE_MAPS_PATH = RUN_DIR / "feature_maps.png"
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


def load_or_create_selected_classes(base_train_dataset):
    if SELECTED_CLASSES_PATH.exists():
        with open(SELECTED_CLASSES_PATH, "r", encoding="utf-8") as f:
            selected_classes = json.load(f)

        print(f"Seçili sınıflar dosyadan yüklendi: {SELECTED_CLASSES_PATH}")
        return selected_classes

    all_classes = base_train_dataset.classes
    selected_classes = random.sample(all_classes, NUM_CLASSES)
    selected_classes = sorted(selected_classes)

    with open(SELECTED_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_classes, f, indent=4, ensure_ascii=False)

    print(f"Seçili sınıflar oluşturuldu ve kaydedildi: {SELECTED_CLASSES_PATH}")
    return selected_classes


def prepare_dataloaders():
    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)
    base_test_dataset = datasets.ImageFolder(TEST_DIR)

    selected_classes = load_or_create_selected_classes(base_train_dataset)

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
# MODEL BLOCKS
# -------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Kanal dikkat mekanizmasıdır.
    Model, hangi feature kanallarının daha önemli olduğunu öğrenir.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)

        return x * scale


class ResidualSEBlock(nn.Module):
    """
    Residual + SE block.

    Bu blokta:
    - 3x3 convolution kullanılır.
    - BatchNorm ile eğitim stabilize edilir.
    - Skip connection ile gradient akışı iyileştirilir.
    - SE block ile kanal bazlı attention eklenir.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.se = SEBlock(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)

        out = out + identity
        out = self.relu(out)

        return out


# -------------------------
# PROPOSED MODEL: CARNET V2
# -------------------------

class CarNet(nn.Module):
    """
    Modified AlexNet-inspired CNN.

    Baseline AlexNet'ten farkları:
    1. İlk 11x11 stride 4 convolution yerine küçük 3x3 convolution blokları.
    2. Conv katmanlarından sonra Batch Normalization.
    3. Residual bağlantılar.
    4. SE kanal dikkat mekanizması.
    5. Büyük fully connected yapı yerine Global Average Pooling.
    """
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            ResidualSEBlock(32, 64, stride=1),
            ResidualSEBlock(64, 64, stride=1),
        )

        self.stage2 = nn.Sequential(
            ResidualSEBlock(64, 128, stride=2),
            ResidualSEBlock(128, 128, stride=1),
        )

        self.stage3 = nn.Sequential(
            ResidualSEBlock(128, 256, stride=2),
            ResidualSEBlock(256, 256, stride=1),
        )

        self.stage4 = nn.Sequential(
            ResidualSEBlock(256, 384, stride=2),
            ResidualSEBlock(384, 384, stride=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x


# -------------------------
# TRAIN / EVAL
# -------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, desc="Training", leave=False)

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
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Augmentation örnekleri kaydedildi: {output_path}")

def save_model_summary(model, output_path):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = []
    lines.append("MODEL SUMMARY - CARNET V2")
    lines.append("=" * 100)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {trainable_params:,}")
    lines.append("=" * 100)
    lines.append("")

    for name, module in model.named_modules():
        if name == "":
            continue

        lines.append(f"Layer Name : {name}")
        lines.append(f"Layer Type : {module.__class__.__name__}")

        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        lines.append(f"Params     : {params:,}")
        lines.append(f"Trainable  : {trainable:,}")
        lines.append("-" * 100)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Model summary kaydedildi: {output_path}")


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
    """
    CarNet v2'de ilk conv katmanı:
    model.stem[0].block[0]
    """
    conv_layer = model.stem[0].block[0]
    weights = conv_layer.weight.data

    num_filters = min(max_filters, weights.shape[0])
    cols = 4
    rows = int(np.ceil(num_filters / cols))

    plt.figure(figsize=(10, 10))

    for i in range(num_filters):
        filt = normalize_filter_image(weights[i])

        plt.subplot(rows, cols, i + 1)
        plt.imshow(filt)
        plt.title(f"Filter {i}")
        plt.axis("off")

    plt.suptitle("CarNet v2 - First Convolution Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"İlk convolution filtreleri kaydedildi: {output_path}")


def save_feature_maps(model, selected_classes, output_path, device):
    """
    Stem, stage2 ve stage4 çıktılarından örnek feature map üretir.
    """
    model.eval()

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

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = eval_transform(original_image).unsqueeze(0).to(device)

    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    hooks.append(model.stem.register_forward_hook(get_hook("stem")))
    hooks.append(model.stage2.register_forward_hook(get_hook("stage2")))
    hooks.append(model.stage4.register_forward_hook(get_hook("stage4")))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    layer_names = ["stem", "stage2", "stage4"]

    plt.figure(figsize=(16, 10))

    plot_index = 1
    for row_idx, layer_name in enumerate(layer_names):
        feature_maps = activations[layer_name][0]  # [C, H, W]
        num_channels = min(8, feature_maps.shape[0])

        for col_idx in range(8):
            plt.subplot(3, 8, plot_index)

            if col_idx < num_channels:
                fmap = feature_maps[col_idx].numpy()
                plt.imshow(fmap, cmap="viridis")
                plt.title(f"{layer_name}\nch {col_idx}", fontsize=8)

            plt.axis("off")
            plot_index += 1

    plt.suptitle(
        f"CarNet v2 Feature Maps\nSample Class: {class_name}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Feature map görselleri kaydedildi: {output_path}")

def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation="nearest")
    plt.title("CarNet v2 Confusion Matrix")
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
    model = CarNet(num_classes=NUM_CLASSES).to(device)
    save_model_summary(model, MODEL_SUMMARY_PATH)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

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
                "model_name": "CarNet v2",
                "run_id": RUN_ID,
            }

            torch.save(checkpoint_data, BEST_MODEL_PATH)

            # Eski sabit path'i de güncel tutuyoruz.
            torch.save(checkpoint_data, LEGACY_BEST_MODEL_PATH)

            print(f"Yeni en iyi CarNet v2 modeli kaydedildi: {BEST_MODEL_PATH}")
            print(f"Legacy model yolu güncellendi: {LEGACY_BEST_MODEL_PATH}")
            print()

    plot_history(history)
    save_history(history)

    print("En iyi CarNet v2 modeli test için yükleniyor...")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_first_conv_filters(model, FIRST_CONV_FILTERS_PATH)
    save_feature_maps(
        model=model,
        selected_classes=selected_classes,
        output_path=FEATURE_MAPS_PATH,
        device=device,
    )

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    print()
    print("CARNET V2 TEST SONUÇLARI")
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
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
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
        "augmentation_examples_path": str(AUGMENTATION_EXAMPLES_PATH),
        "model_summary_path": str(MODEL_SUMMARY_PATH),
        "first_conv_filters_path": str(FIRST_CONV_FILTERS_PATH),
        "feature_maps_path": str(FEATURE_MAPS_PATH),
    }

    save_metrics(metrics)
    append_metrics_to_csv(metrics)

    print()
    print("Çalıştırma tamamlandı.")
    print(f"Tüm çıktı klasörü: {RUN_DIR}")


if __name__ == "__main__":
    main()