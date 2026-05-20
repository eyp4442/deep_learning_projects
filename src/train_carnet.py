import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

TRAIN_DIR = Path("data/stanford_cars/train")
TEST_DIR = Path("data/stanford_cars/test")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES_PATH = OUTPUT_DIR / "selected_classes_30.json"
BEST_MODEL_PATH = MODEL_DIR / "carnet30_best.pth"
REPORT_PATH = OUTPUT_DIR / "carnet_classification_report.txt"
CURVE_PATH = FIGURE_DIR / "carnet_training_curves.png"


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
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
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
# PROPOSED MODEL: CARNET
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


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CarNet Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURVE_PATH)
    plt.close()

    print(f"Grafik kaydedildi: {CURVE_PATH}")


# -------------------------
# MAIN
# -------------------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader, selected_classes = prepare_dataloaders()

    model = CarNet(num_classes=NUM_CLASSES).to(device)

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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
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

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 70)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_classes": selected_classes,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "model_name": "CarNet",
                },
                BEST_MODEL_PATH,
            )

            print(f"Yeni en iyi CarNet modeli kaydedildi: {BEST_MODEL_PATH}")
            print()

    plot_history(history)

    print("En iyi CarNet modeli test için yükleniyor...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    print()
    print("CARNET TEST SONUÇLARI")
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


if __name__ == "__main__":
    main()