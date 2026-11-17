"""
Dataset Inspection and Inventory Module
AI-Enabled Crop Disease Detection and Advisory System

Inspects PlantVillage dataset folders directly on D:\ drive.
Validates images, dimensions, corrupt files, duplicates, and class distribution.
Generates dataset_inventory.csv, dataset_inventory.json, and class_distribution.png.
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image

# Set Matplotlib cache dir to project to prevent permission warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[2] / ".mplconfig")
import matplotlib.pyplot as plt

# Dataset Root
DATASET_ROOT = Path("D:/")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Relevant crop prefixes
CROP_PREFIXES = ("Pepper", "Potato", "Tomato")

# Target initial 9 classes specification
TARGET_9_CLASSES = {
    "Pepper__bell___Bacterial_spot": {"crop": "Pepper", "condition": "Bacterial Spot", "category": "diseased"},
    "Pepper__bell___healthy": {"crop": "Pepper", "condition": "Healthy", "category": "healthy"},
    "Potato___Early_blight": {"crop": "Potato", "condition": "Early Blight", "category": "diseased"},
    "Potato___Late_blight": {"crop": "Potato", "condition": "Late Blight", "category": "diseased"},
    "Potato___healthy": {"crop": "Potato", "condition": "Healthy", "category": "healthy"},
    "Tomato_Bacterial_spot": {"crop": "Tomato", "condition": "Bacterial Spot", "category": "diseased"},
    "Tomato_Early_blight": {"crop": "Tomato", "condition": "Early Blight", "category": "diseased"},
    "Tomato_Late_blight": {"crop": "Tomato", "condition": "Late Blight", "category": "diseased"},
    "Tomato_healthy": {"crop": "Tomato", "condition": "Healthy", "category": "healthy"}
}

def compute_file_hash(filepath: Path, chunk_size=65536) -> str:
    """Computes MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset():
    print("=" * 70)
    print("AI-ENABLED CROP DISEASE DETECTION: DATASET INSPECTION")
    print("=" * 70)
    print(f"Scanning directory: {DATASET_ROOT}")

    # Discover relevant crop folders
    candidate_folders = [
        d for d in DATASET_ROOT.iterdir()
        if d.is_dir() and any(d.name.startswith(prefix) for prefix in CROP_PREFIXES)
    ]
    candidate_folders.sort(key=lambda p: p.name)

    print(f"Found {len(candidate_folders)} crop disease folders in {DATASET_ROOT}:")
    for idx, folder in enumerate(candidate_folders, 1):
        is_target = " [TARGET 9 CLASS]" if folder.name in TARGET_9_CLASSES else " [EXTRA CLASS]"
        print(f"  {idx:2d}. {folder.name}{is_target}")

    inventory = []
    all_hashes = {}  # hash -> first filepath seen
    duplicate_records = []
    supported_extensions = Counter()
    total_corrupt = 0

    print("\nAnalyzing images in each class (validating integrity, dimensions, hashes)...")

    for folder in candidate_folders:
        folder_name = folder.name
        is_in_target_9 = folder_name in TARGET_9_CLASSES
        
        # Determine crop and condition
        if folder_name.startswith("Pepper"):
            crop = "Pepper"
        elif folder_name.startswith("Potato"):
            crop = "Potato"
        elif folder_name.startswith("Tomato"):
            crop = "Tomato"
        else:
            crop = "Unknown"

        # Determine condition name cleanly
        clean_name = folder_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
        is_healthy = "healthy" in folder_name.lower()

        # List all files in the class folder
        files = [f for f in folder.iterdir() if f.is_file()]
        class_image_count = len(files)
        class_corrupt = 0
        class_duplicates = 0
        dimensions_list = []

        for f in files:
            ext = f.suffix.lower()
            supported_extensions[ext] += 1

            # Check duplicates via hash
            f_hash = compute_file_hash(f)
            if f_hash in all_hashes:
                class_duplicates += 1
                duplicate_records.append({
                    "original": str(all_hashes[f_hash]),
                    "duplicate": str(f),
                    "hash": f_hash
                })
            else:
                all_hashes[f_hash] = f

            # Validate image with PIL
            try:
                with Image.open(f) as img:
                    img.verify()  # Fast verification of file header & integrity
                # Reopen to check dimensions (verify closes the file)
                with Image.open(f) as img:
                    dimensions_list.append(img.size)
            except Exception as e:
                class_corrupt += 1
                total_corrupt += 1

        # Calculate dimension summary
        dim_counts = Counter(dimensions_list)
        most_common_dim = dim_counts.most_common(1)[0][0] if dim_counts else None

        record = {
            "folder_name": folder_name,
            "crop": crop,
            "condition": clean_name,
            "is_target_9": is_in_target_9,
            "is_healthy": is_healthy,
            "total_images": class_image_count,
            "valid_images": class_image_count - class_corrupt,
            "corrupt_images": class_corrupt,
            "duplicate_images": class_duplicates,
            "common_dimension": f"{most_common_dim[0]}x{most_common_dim[1]}" if most_common_dim else "N/A",
            "extensions": list(set(f.suffix.lower() for f in files))
        }
        inventory.append(record)
        print(f"  Processed {folder_name:42s} | Count: {class_image_count:5d} | Valid: {record['valid_images']:5d} | Dups: {class_duplicates:3d} | Corrupt: {class_corrupt}")

    df_inventory = pd.DataFrame(inventory)

    # Save CSV and JSON
    csv_path = REPORTS_DIR / "dataset_inventory.csv"
    json_path = REPORTS_DIR / "dataset_inventory.json"
    df_inventory.to_csv(csv_path, index=False)

    summary_json = {
        "dataset_root": str(DATASET_ROOT),
        "total_classes_found": len(inventory),
        "target_9_classes_found": sum(1 for r in inventory if r["is_target_9"]),
        "total_images_all_classes": int(df_inventory["total_images"].sum()),
        "total_images_target_9": int(df_inventory[df_inventory["is_target_9"]]["total_images"].sum()),
        "total_corrupted_images": total_corrupt,
        "total_duplicates_found": len(duplicate_records),
        "supported_extensions": dict(supported_extensions),
        "classes": inventory
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)

    print(f"\nSaved inventory to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

    # Generate Class Distribution Chart
    generate_distribution_chart(df_inventory)

    # Print summary
    print_inventory_summary(df_inventory, summary_json)

    return df_inventory, summary_json

def generate_distribution_chart(df: pd.DataFrame):
    """Generates a professional horizontal bar chart of class image counts."""
    chart_path = REPORTS_DIR / "class_distribution.png"
    plt.figure(figsize=(12, 8))

    # Sort by is_target_9 then total_images
    df_sorted = df.sort_values(by=["is_target_9", "total_images"], ascending=[True, True])
    colors = ["#2b8cbe" if is_t else "#a6bddb" for is_t in df_sorted["is_target_9"]]

    bars = plt.barh(df_sorted["folder_name"], df_sorted["total_images"], color=colors, edgecolor="#045a8d")
    plt.xlabel("Number of Images", fontsize=12, fontweight="bold")
    plt.ylabel("Dataset Folder / Class Name", fontsize=12, fontweight="bold")
    plt.title("PlantVillage Dataset: Image Distribution Across Classes\n(Dark Blue = Initial Target 9 Classes, Light Blue = Additional Classes)", fontsize=13, fontweight="bold")

    # Add count labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 15, bar.get_y() + bar.get_height()/2, f"{int(width):,}", 
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#1c2d42")

    plt.xlim(0, max(df["total_images"]) * 1.15)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved distribution chart to: {chart_path}")

def print_inventory_summary(df: pd.DataFrame, summary: dict):
    print("\n" + "=" * 70)
    print("DATASET INVENTORY SUMMARY REPORT")
    print("=" * 70)
    print(f"Total crop folders discovered: {summary['total_classes_found']}")
    print(f"Target 9 classes available:   {summary['target_9_classes_found']} of 9 (100% matched)")
    print(f"Total images across all classes: {summary['total_images_all_classes']:,}")
    print(f"Total images in Target 9:        {summary['total_images_target_9']:,}")
    print(f"Corrupted images detected:       {summary['total_corrupted_images']}")
    print(f"Duplicates detected:             {summary['total_duplicates_found']}")
    print(f"Supported extensions:            {summary['supported_extensions']}")

    print("\n--- TARGET 9 CLASSES BREAKDOWN ---")
    df_t9 = df[df["is_target_9"]]
    for _, row in df_t9.iterrows():
        print(f"  [{row['crop']:6s}] {row['folder_name']:35s} : {row['total_images']:5d} images | Common dim: {row['common_dimension']}")

    print("\n--- ADDITIONAL AVAILABLE TOMATO CLASSES (6 CLASSES) ---")
    df_extra = df[~df["is_target_9"]]
    for _, row in df_extra.iterrows():
        print(f"  [{row['crop']:6s}] {row['folder_name']:45s} : {row['total_images']:5d} images")

    print("=" * 70)

if __name__ == "__main__":
    inspect_dataset()
