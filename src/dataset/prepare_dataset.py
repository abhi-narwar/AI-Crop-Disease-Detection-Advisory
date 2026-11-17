"""
Dataset Preparation and Stratified Splitting Module
AI-Enabled Crop Disease Detection and Advisory System

Splits all 15 classes from D:\ into:
- 70% Training
- 15% Validation
- 15% Testing

Features:
- Stratified splitting per class with reproducible seed (42).
- Filters out corrupted images and exact duplicate images to prevent data leakage.
- Generates data/manifests/{train,val,test}.csv
- Generates data/processed/{train,val,test}/<class_name>/ image structure.
- Saves models/class_indices.json and reports/split_summary.json.
- Generates reports/split_class_distribution.png.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path("D:/")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib cache
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
import matplotlib.pyplot as plt

RANDOM_SEED = 42

# 15 Approved Classes (Sorted alphabetically for deterministic indexing)
ALL_15_CLASSES = sorted([
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
])

def compute_file_hash(filepath: Path, chunk_size=65536) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_class_metadata(folder_name: str, class_idx: int):
    """Extract crop, readable disease condition, and healthy status."""
    if folder_name.startswith("Pepper"):
        crop = "Pepper"
    elif folder_name.startswith("Potato"):
        crop = "Potato"
    else:
        crop = "Tomato"

    is_healthy = "healthy" in folder_name.lower()
    
    # Human-friendly condition name
    readable = (folder_name
                .replace("___", " - ")
                .replace("__", " - ")
                .replace("_", " "))
    # Remove leading crop from condition if repeated
    if readable.startswith(crop):
        condition = readable[len(crop):].strip(" -")
    else:
        condition = readable

    return {
        "class_index": class_idx,
        "class_name": folder_name,
        "crop": crop,
        "condition": condition if condition else "Healthy",
        "is_healthy": is_healthy
    }

def prepare_dataset(copy_images=True):
    print("=" * 70)
    print("AI-ENABLED CROP DISEASE DETECTION: DATASET PREPARATION & SPLITTING")
    print(f"Seed: {RANDOM_SEED} | Ratio: 70% Train, 15% Val, 15% Test")
    print("=" * 70)

    # Build and save class indices mapping
    class_indices = {}
    class_metadata = {}
    for idx, cname in enumerate(ALL_15_CLASSES):
        class_indices[cname] = idx
        class_metadata[cname] = clean_class_metadata(cname, idx)

    indices_path = MODELS_DIR / "class_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved class indices mapping to {indices_path}")

    # Discover and validate images, filtering duplicates and corruption
    valid_records_by_class = defaultdict(list)
    seen_hashes = set()
    total_corrupt = 0
    total_duplicates = 0

    print("\nValidating and filtering images per class...")
    for cname in ALL_15_CLASSES:
        folder = DATASET_ROOT / cname
        if not folder.exists():
            raise FileNotFoundError(f"Expected class folder does not exist: {folder}")

        files = [f for f in folder.iterdir() if f.is_file()]
        for f in files:
            # Check hash for exact duplicate
            f_hash = compute_file_hash(f)
            if f_hash in seen_hashes:
                total_duplicates += 1
                continue
            seen_hashes.add(f_hash)

            # Check image integrity
            try:
                with Image.open(f) as img:
                    img.verify()
                valid_records_by_class[cname].append(f)
            except Exception:
                total_corrupt += 1

        print(f"  [{cname}] Valid unique images: {len(valid_records_by_class[cname])}")

    print(f"\nTotal unique valid images across all 15 classes: {len(seen_hashes)}")
    print(f"Excluded exact duplicates: {total_duplicates} | Excluded corrupt: {total_corrupt}")

    # Stratified Splitting: 70% Train, 15% Val, 15% Test
    rng = np.random.RandomState(RANDOM_SEED)

    train_data = []
    val_data = []
    test_data = []
    split_summary = {"classes": {}, "totals": {"train": 0, "val": 0, "test": 0, "total": 0}}

    for cname in ALL_15_CLASSES:
        file_list = valid_records_by_class[cname]
        n_total = len(file_list)

        # Permute deterministically
        indices = rng.permutation(n_total)
        shuffled_files = [file_list[i] for i in indices]

        n_train = int(round(0.70 * n_total))
        n_val = int(round(0.15 * n_total))
        n_test = n_total - n_train - n_val

        c_train = shuffled_files[:n_train]
        c_val = shuffled_files[n_train:n_train + n_val]
        c_test = shuffled_files[n_train + n_val:]

        meta = class_metadata[cname]

        for f in c_train:
            train_data.append({
                "filepath": str(f.resolve()),
                "filename": f.name,
                "class_name": cname,
                "class_index": meta["class_index"],
                "crop": meta["crop"],
                "condition": meta["condition"],
                "is_healthy": meta["is_healthy"]
            })
        for f in c_val:
            val_data.append({
                "filepath": str(f.resolve()),
                "filename": f.name,
                "class_name": cname,
                "class_index": meta["class_index"],
                "crop": meta["crop"],
                "condition": meta["condition"],
                "is_healthy": meta["is_healthy"]
            })
        for f in c_test:
            test_data.append({
                "filepath": str(f.resolve()),
                "filename": f.name,
                "class_name": cname,
                "class_index": meta["class_index"],
                "crop": meta["crop"],
                "condition": meta["condition"],
                "is_healthy": meta["is_healthy"]
            })

        split_summary["classes"][cname] = {
            "total": n_total,
            "train": len(c_train),
            "val": len(c_val),
            "test": len(c_test),
            "train_pct": round(len(c_train) / n_total * 100, 2),
            "val_pct": round(len(c_val) / n_total * 100, 2),
            "test_pct": round(len(c_test) / n_total * 100, 2)
        }

        split_summary["totals"]["train"] += len(c_train)
        split_summary["totals"]["val"] += len(c_val)
        split_summary["totals"]["test"] += len(c_test)
        split_summary["totals"]["total"] += n_total

    # Convert to DataFrames and save manifests
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)

    df_train.to_csv(MANIFESTS_DIR / "train.csv", index=False)
    df_val.to_csv(MANIFESTS_DIR / "val.csv", index=False)
    df_test.to_csv(MANIFESTS_DIR / "test.csv", index=False)

    print(f"\nManifests saved to {MANIFESTS_DIR}:")
    print(f"  train.csv: {len(df_train)} rows ({split_summary['totals']['train'] / split_summary['totals']['total'] * 100:.2f}%)")
    print(f"  val.csv:   {len(df_val)} rows ({split_summary['totals']['val'] / split_summary['totals']['total'] * 100:.2f}%)")
    print(f"  test.csv:  {len(df_test)} rows ({split_summary['totals']['test'] / split_summary['totals']['total'] * 100:.2f}%)")

    # Save summary JSON
    with open(REPORTS_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=4)
    print(f"Saved split summary to {REPORTS_DIR / 'split_summary.json'}")

    # Generate split distribution plot
    generate_split_plot(split_summary)

    # Populate data/processed/{train,val,test}/<class>
    if copy_images:
        print("\nCreating organized dataset in data/processed/...")
        for split_name, df_split in [("train", df_train), ("val", df_split_name := df_val), ("test", df_test)]:
            print(f"  Populating data/processed/{split_name}...")
            for _, row in df_split.iterrows():
                dest_dir = PROCESSED_DIR / split_name / row["class_name"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / row["filename"]
                if not dest_file.exists():
                    shutil.copy2(row["filepath"], dest_file)
        print("Dataset successfully organized in data/processed/!")

    print("=" * 70)
    print("PHASE 2 DATASET PREPARATION COMPLETE!")
    print("=" * 70)

def generate_split_plot(summary: dict):
    classes = list(summary["classes"].keys())
    train_counts = [summary["classes"][c]["train"] for c in classes]
    val_counts = [summary["classes"][c]["val"] for c in classes]
    test_counts = [summary["classes"][c]["test"] for c in classes]

    y = np.arange(len(classes))
    height = 0.25

    plt.figure(figsize=(14, 10))
    plt.barh(y - height, train_counts, height, label="Train (70%)", color="#2ca02c")
    plt.barh(y, val_counts, height, label="Validation (15%)", color="#1f77b4")
    plt.barh(y + height, test_counts, height, label="Test (15%)", color="#d62728")

    plt.yticks(y, classes, fontsize=9)
    plt.xlabel("Number of Images", fontsize=11, fontweight="bold")
    plt.title("Stratified 70 : 15 : 15 Split Across All 15 Classes (No Data Leakage)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    chart_path = REPORTS_DIR / "split_class_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved split distribution chart to: {chart_path}")

if __name__ == "__main__":
    prepare_dataset(copy_images=True)
