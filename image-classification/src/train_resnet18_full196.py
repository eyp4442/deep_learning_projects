import random
from pathlib import Path
from collections import defaultdict

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
IMG_SIZE = 224
BATCH_SIZE = 16

# Full 196 sınıf daha uzun sürecek.
# Önce bu ayarlarla deneyelim. Gerekirse sonra artırırız.
EPOCHS_STAGE1 = 3
LR_STAGE1 = 1e-3

EPOCHS_STAGE2 = 6
LR_STAGE2 = 1e-4

WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.20

TRAIN_DIR = Path("data/stanford_cars/train")
TEST_DIR = Path("data/stanford_cars/test")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "resnet18_full196_best.pth"
REPORT_PATH = OUTPUT_DIR / "resnet18_full196_classification_report.txt"
CURVE_PATH = FIGURE_DIR / "resnet18_full196_training_curves.png"


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
    """
    Her sınıftan yaklaşık aynı oranda validation ayırır.
    Full 196 sınıfta rastgele global split yerine bu daha doğru.
    """
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

    train_dataset = StanfordCarsDataset(train_samples, transform=train_transform)
    val_dataset = StanfordCarsDataset(val_samples, transform=eval_transform)
    test_dataset = StanfordCarsDataset(test_samples, transform=eval_transform)

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

    # Başta bütün pretrained backbone dondurulur.
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
    # Fine-tuning aşamasında son ResNet bloğu ve classifier açılır.
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

def top5_accuracy(outputs, labels):
    _, top5_preds = outputs.topk(5, dim=1)
    correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
    return correct.any(dim=1).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, desc):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    top5_scores = []

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
        top5_scores.append(top5_accuracy(outputs.detach(), labels.detach()))

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_top5 = float(np.mean(top5_scores))

    return epoch_loss, epoch_acc, epoch_top5


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    top5_scores = []

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
            top5_scores.append(top5_accuracy(outputs.detach(), labels.detach()))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_top5 = float(np.mean(top5_scores))

    return epoch_loss, epoch_acc, epoch_f1, epoch_top5, all_labels, all_preds


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Full 196 Training and Validation Loss")
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_top5"].append(train_top5)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_top5"].append(val_top5)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top-5: {train_top5:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | Val Top-5: {val_top5:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 90)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_classes": selected_classes,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_top5": val_top5,
                    "model_name": "ResNet18 Full 196 Transfer Learning",
                },
                BEST_MODEL_PATH,
            )

            print(f"Yeni en iyi ResNet18 Full196 modeli kaydedildi: {BEST_MODEL_PATH}")
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

    num_classes = len(selected_classes)
    model = build_resnet18_transfer(num_classes=num_classes).to(device)

    total_params, trainable_params = count_parameters(model)

    print("MODEL: ResNet18 Transfer Learning Full 196")
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

    print("En iyi ResNet18 Full196 modeli test için yükleniyor...")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_top5, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print("RESNET18 FULL 196 TRANSFER LEARNING TEST SONUÇLARI")
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


if __name__ == "__main__":
    main()