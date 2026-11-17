"""
Automated Unit Tests for Dataset Preparation and Splitting
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

def test_class_indices_file():
    indices_file = MODELS_DIR / "class_indices.json"
    assert indices_file.exists(), "class_indices.json must exist"
    
    with open(indices_file, "r") as f:
        indices = json.load(f)
    assert len(indices) == 15, "Must track exactly 15 classes"
    assert "Tomato_healthy" in indices
    assert "Pepper__bell___Bacterial_spot" in indices
    assert "Potato___Early_blight" in indices

def test_manifest_files_exist():
    for name in ["train.csv", "val.csv", "test.csv"]:
        path = MANIFESTS_DIR / name
        assert path.exists(), f"Manifest {name} must exist"

def test_split_proportions():
    df_train = pd.read_csv(MANIFESTS_DIR / "train.csv")
    df_val = pd.read_csv(MANIFESTS_DIR / "val.csv")
    df_test = pd.read_csv(MANIFESTS_DIR / "test.csv")

    total = len(df_train) + len(df_val) + len(df_test)
    assert total > 20000, f"Expected total images > 20000, got {total}"

    train_pct = len(df_train) / total * 100
    val_pct = len(df_val) / total * 100
    test_pct = len(df_test) / total * 100

    # Verify within 0.5% of 70:15:15 target
    assert 69.5 <= train_pct <= 70.5, f"Train % was {train_pct:.2f}%"
    assert 14.5 <= val_pct <= 15.5, f"Val % was {val_pct:.2f}%"
    assert 14.5 <= test_pct <= 15.5, f"Test % was {test_pct:.2f}%"

def test_zero_data_leakage():
    df_train = pd.read_csv(MANIFESTS_DIR / "train.csv")
    df_val = pd.read_csv(MANIFESTS_DIR / "val.csv")
    df_test = pd.read_csv(MANIFESTS_DIR / "test.csv")

    train_files = set(df_train["filepath"])
    val_files = set(df_val["filepath"])
    test_files = set(df_test["filepath"])

    assert len(train_files.intersection(val_files)) == 0, "Data leakage between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Data leakage between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Data leakage between val and test!"
