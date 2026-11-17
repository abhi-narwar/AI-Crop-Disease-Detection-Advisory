"""
MobileNetV2 Model Training Pipeline
AI-Enabled Crop Disease Detection and Advisory System

Features:
- Loads training (14,438 images) and validation (3,095 images) via tf.data with prefetching.
- Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomContrast).
- MobileNetV2 Transfer Learning with ImageNet weights.
- Two-Stage Training:
    Stage 1: Frozen Base Training (Classification Head)
    Stage 2: Fine-Tuning top convolutional layers with lower learning rate.
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
- Artifacts: crop_model.keras, model_metadata.json, training_history.json, training_accuracy.png, training_loss.png.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib cache
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
import matplotlib.pyplot as plt

from src.model.build_model import create_crop_disease_model, setup_finetuning, INPUT_SHAPE, IMG_SIZE

# Training Hyperparameters optimized for 8 GB RAM / 4 GB VRAM
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 6
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42

def build_augmentation_layer():
    """Builds standard data augmentation pipeline."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.15, seed=SEED),
        layers.RandomZoom(0.15, seed=SEED),
        layers.RandomTranslation(0.1, 0.1, seed=SEED),
    ], name="data_augmentation")

def load_datasets(batch_size=BATCH_SIZE):
    """Loads tf.data datasets from data/processed directory."""
    print(f"Loading training data from: {TRAIN_DIR}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=SEED
    )

    print(f"Loading validation data from: {VAL_DIR}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Verified {len(class_names)} classes in training dataset.")

    # Data augmentation and MobileNetV2 preprocessing
    augmentation = build_augmentation_layer()

    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def train_model(initial_epochs=INITIAL_EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS):
    start_time = time.time()
    print("=" * 70)
    print("AI-ENABLED CROP DISEASE DETECTION: MOBILENETV2 TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load datasets
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    # Ensure class indices match
    class_indices = {cname: idx for idx, cname in enumerate(class_names)}
    with open(MODELS_DIR / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=4)

    # 2. Build MobileNetV2 Model
    print(f"\nConstructing MobileNetV2 model for {num_classes} classes...")
    model, base_model = create_crop_disease_model(num_classes=num_classes, learning_rate=INITIAL_LR)
    model.summary()

    # 3. Setup Callbacks
    best_model_path = MODELS_DIR / "crop_model.keras"
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Stage 1: Feature Extraction (Frozen Base)
    print("\n" + "=" * 70)
    print(f"STAGE 1: Training Classification Head ({initial_epochs} epochs, lr={INITIAL_LR})")
    print("=" * 70)
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    # 5. Stage 2: Fine-Tuning Top Layers
    total_epochs = initial_epochs + fine_tune_epochs
    print("\n" + "=" * 70)
    print(f"STAGE 2: Fine-Tuning MobileNetV2 Top Layers ({fine_tune_epochs} epochs, lr={FINE_TUNE_LR})")
    print("=" * 70)
    model = setup_finetuning(model, base_model, unfreeze_from_layer=120, fine_tune_lr=FINE_TUNE_LR)

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_stage1.history["loss"]),
        epochs=total_epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    # Combine History
    combined_history = {}
    for k in history_stage1.history.keys():
        combined_history[k] = [float(v) for v in history_stage1.history[k] + history_stage2.history.get(k, [])]

    # Save training history JSON
    history_path = REPORTS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\nSaved training history to {history_path}")

    # Generate Training Accuracy & Loss Graphs
    plot_training_metrics(combined_history)

    # Save Model Metadata
    elapsed_time = round(time.time() - start_time, 2)
    best_val_acc = max(combined_history.get("val_accuracy", [0.0]))
    best_val_loss = min(combined_history.get("val_loss", [0.0]))

    metadata = {
        "model_architecture": "MobileNetV2",
        "num_classes": num_classes,
        "input_shape": list(INPUT_SHAPE),
        "batch_size": BATCH_SIZE,
        "initial_epochs": initial_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "total_trained_epochs": len(combined_history["loss"]),
        "training_duration_seconds": elapsed_time,
        "best_val_accuracy": round(float(best_val_acc) * 100, 2),
        "best_val_loss": round(float(best_val_loss), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {MODELS_DIR / 'model_metadata.json'}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Duration: {elapsed_time/60:.2f} mins")
    print(f"Best Validation Accuracy: {metadata['best_val_accuracy']}%")
    print(f"Best Validation Loss:     {metadata['best_val_loss']}")
    print(f"Best model saved to:      {best_model_path}")
    print("=" * 70)

    return model, combined_history

def plot_training_metrics(history: dict):
    epochs = range(1, len(history["loss"]) + 1)

    # 1. Accuracy Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [a * 100 for a in history["accuracy"]], "g-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_accuracy"]], "b-s", label="Validation Accuracy", linewidth=2)
    plt.title("MobileNetV2: Training vs. Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_path = REPORTS_DIR / "training_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved training accuracy chart to {acc_path}")

    # 2. Loss Plot
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["loss"], "r-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "m-s", label="Validation Loss", linewidth=2)
    plt.title("MobileNetV2: Training vs. Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss (Categorical Cross-Entropy)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = REPORTS_DIR / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved training loss chart to {loss_path}")

if __name__ == "__main__":
    train_model()
