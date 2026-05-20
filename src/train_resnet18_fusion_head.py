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
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


# -------------------------
# CONFIG
# -------------------------

SEED = 42
NUM_CLASSES = 30
IMG_SIZE = 224
BATCH_SIZE = 16

# Stage 1: sadece son classifier eğitilecek
EPOCHS_STAGE1 = 5
LR_STAGE1 = 1e-3

# Stage 2: ResNet'in son bloğu da açılacak
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

BEST_MODEL_PATH = MODEL_DIR / "resnet18_fusion_head_30_best.pth"
REPORT_PATH = OUTPUT_DIR / "resnet18_fusion_head_classification_report.txt"
CURVE_PATH = FIGURE_DIR / "resnet18_fusion_head_training_curves.png"


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
            "Önce baseline/CarNet dosyalarını çalıştırmış olman gerekiyor."
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

class ResNet18FusionHead(nn.Module):
    """
    ResNet18 tabanlı özelleştirilmiş transfer learning modeli.

    Standart ResNet18'den farkları:
    1. Son fc katmanı kaldırıldı.
    2. Feature map üzerinde SE-like channel attention eklendi.
    3. Global Average Pooling + Global Max Pooling birlikte kullanıldı.
    4. Daha güçlü özel classifier eklendi.
    """
    def __init__(self, num_classes):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)

        # ResNet18'in fc ve avgpool öncesine kadar olan convolutional kısmını alıyoruz.
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Başlangıçta backbone dondurulur.
        for param in self.features.parameters():
            param.requires_grad = False

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 512),
            nn.Sigmoid(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.45),
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)

        attention = self.channel_attention(x).view(x.size(0), 512, 1, 1)
        x = x * attention

        avg_features = self.global_avg_pool(x)
        max_features = self.global_max_pool(x)

        x = torch.cat([avg_features, max_features], dim=1)
        x = self.classifier(x)

        return x


def build_resnet18_transfer(num_classes):
    return ResNet18FusionHead(num_classes)


def unfreeze_layer4(model):
    # self.features içinde layer4 son bloktur.
    # ResNet children:
    # 0 conv1, 1 bn1, 2 relu, 3 maxpool, 4 layer1, 5 layer2, 6 layer3, 7 layer4
    for param in model.features[7].parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.channel_attention.parameters():
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


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Transfer Learning Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURVE_PATH)
    plt.close()

    print(f"Grafik kaydedildi: {CURVE_PATH}")


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
                    "model_name": "ResNet18 Fusion Head Transfer Learning",
                },
                BEST_MODEL_PATH,
            )

            print(f"Yeni en iyi ResNet18 modeli kaydedildi: {BEST_MODEL_PATH}")
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

    train_loader, val_loader, test_loader, selected_classes = prepare_dataloaders()

    model = build_resnet18_transfer(num_classes=NUM_CLASSES).to(device)

    total_params, trainable_params = count_parameters(model)

    print("MODEL: ResNet18 Fusion Head Transfer Learning")
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

    print("En iyi ResNet18 modeli test için yükleniyor...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

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


if __name__ == "__main__":
    main()