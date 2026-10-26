
from pathlib import Path

DATA_DIR = Path("data/raw")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not DATA_DIR.exists():
    print("Dataset folder nahi mila:", DATA_DIR)
else:
    total_images = 0

    print("\nTOMATO DATASET REPORT")
    print("-" * 40)

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        image_count = sum(
            1
            for file_path in class_dir.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_dir.name}: {image_count} images")
        total_images += image_count

    print("-" * 40)
    print("Total images:", total_images)

    if total_images == 0:
        print("Check karo: dataset folders ke andar images hain?")
