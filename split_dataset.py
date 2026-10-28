
from pathlib import Path
import random
import csv

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/splits")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42

# Check whether the dataset folder exists
if not DATA_DIR.exists():
    raise SystemExit(
        "ERROR: data/raw folder not found. Please check the dataset location."
    )

# Find the class folders containing images
class_folders = sorted(
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir()
    and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        for p in folder.rglob("*")
    )
)

if len(class_folders) < 2:
    raise SystemExit(
        "ERROR: At least 2 class folders must contain images."
    )

random.seed(RANDOM_SEED)
splits = {"train": [], "validation": [], "test": []}

print("\nDATASET SPLIT REPORT")
print("-" * 45)

for folder in class_folders:
    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    random.shuffle(images)

    total = len(images)

    if total < 10:
        raise SystemExit(
            f"ERROR: Only {total} images found in {folder.name}. "
            "Please check the dataset and class labels."
        )

    # Split images into 70% training, 15% validation, and 15% testing
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    groups = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:],
    }

    print(f"\n{folder.name}")

    for split_name, paths in groups.items():
        for path in paths:
            splits[split_name].append(
                (str(path.as_posix()), folder.name)
            )

        print(f"  {split_name}: {len(paths)} images")

print("\nTotal images:", sum(len(v) for v in splits.values()))

# Create the output directory
# Images will not be copied; only CSV manifests will be saved
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split_name, rows in splits.items():
    random.shuffle(rows)
    output_file = OUTPUT_DIR / f"{split_name}.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)

    print(f"Saved: {output_file}")

print("\nDataset splitting completed successfully!")
print("Original images remain in their respective data/raw folders.")
