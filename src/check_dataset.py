import os
from pathlib import Path
from torchvision import datasets


BASE_DIR = Path("data/stanford_cars")


def find_train_test_dirs(base_dir: Path):
    train_dir = None
    test_dir = None

    for root, dirs, files in os.walk(base_dir):
        lower_dirs = [d.lower() for d in dirs]

        if "train" in lower_dirs:
            train_dir = Path(root) / dirs[lower_dirs.index("train")]

        if "test" in lower_dirs:
            test_dir = Path(root) / dirs[lower_dirs.index("test")]

    return train_dir, test_dir


if __name__ == "__main__":
    train_dir, test_dir = find_train_test_dirs(BASE_DIR)

    print("Train dir:", train_dir)
    print("Test dir:", test_dir)

    if train_dir is None or test_dir is None:
        print("Train/test klasörleri bulunamadı.")
        print("Mevcut klasör yapısı:")

        for root, dirs, files in os.walk(BASE_DIR):
            print(root)
            print("Dirs:", dirs[:5])
            print("Files:", files[:5])
            print("-" * 60)

        raise SystemExit

    train_dataset = datasets.ImageFolder(train_dir)
    test_dataset = datasets.ImageFolder(test_dir)

    print("Toplam train image:", len(train_dataset))
    print("Toplam test image:", len(test_dataset))
    print("Toplam sınıf:", len(train_dataset.classes))
    print("İlk 10 sınıf:")

    for class_name in train_dataset.classes[:10]:
        print(" -", class_name)