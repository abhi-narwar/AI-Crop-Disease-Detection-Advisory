"""
Script to generate production-grade Google Colab Notebook: notebooks/mobilenetv2_training.ipynb
AI-Enabled Crop Disease Detection and Advisory System
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "mobilenetv2_training.ipynb"

def create_notebook():
    cells = []

    def add_markdown(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Cell 1: Title
    add_markdown("""# AI-Enabled Crop Disease Detection and Advisory System
## MobileNetV2 Transfer Learning & Fine-Tuning Pipeline
**B.Tech CSE (AI/ML) Major Project**

This notebook implements an end-to-end deep learning pipeline for plant leaf disease classification across 15 agricultural crop classes (Tomato, Potato, Pepper) using MobileNetV2 transfer learning on Google Colab (with free GPU acceleration).

### Pipeline Stages:
1. **Hardware & GPU Inspection**
2. **Dataset Setup & Validation** (15 Classes, 70:15:15 Train/Val/Test Split)
3. **Data Augmentation & Input Preprocessing** (MobileNetV2 `preprocess_input`)
4. **Model Architecture Construction** (Pretrained MobileNetV2 + GlobalAveragePooling2D + Dropout + Softmax)
5. **Phase 1: Feature Extraction** (Frozen Base, Adam lr=1e-3)
6. **Phase 2: Fine-Tuning** (Unfreeze top convolutional layers, Adam lr=1e-5)
7. **Comprehensive Evaluation on Held-Out Test Set** (Accuracy, Precision, Recall, F1, Confusion Matrix)
8. **Artifact Export** (`crop_model.keras`, `class_indices.json`, `model_metadata.json`)""")

    # Cell 2: Check GPU
    add_code("""# 1. Hardware Inspection: Verify GPU Acceleration
import tensorflow as tf
import os

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"CUDA GPU Detected: {gpus[0]}")
    !nvidia-smi
else:
    print("WARNING: No GPU detected. Please go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.")""")

    # Cell 3: Imports
    add_code("""# 2. Imports and Environment Setup
import os
import sys
import json
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Libraries imported successfully.")""")

    # Cell 4: Dataset Mounting / Loading
    add_markdown("""### 3. Dataset Loading
You can load the dataset through one of two methods:
- **Option A (Google Drive)**: Upload your `data/processed` folder or zip to Google Drive and mount.
- **Option B (Direct Upload / Kaggle)**: Upload `dataset.zip` directly to the Colab runtime.""")

    add_code("""# 3. Mount Google Drive or Setup Dataset Directory
from google.colab import drive
import zipfile

# Check if running in Google Colab
if 'google.colab' in sys.modules:
    # If you have uploaded a zip file (e.g. dataset.zip) to Google Drive or root:
    DATASET_ZIP = "/content/drive/MyDrive/crop_dataset.zip"
    LOCAL_ZIP = "/content/dataset.zip"
    
    if os.path.exists(DATASET_ZIP):
        print(f"Extracting dataset from Google Drive: {DATASET_ZIP}")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall("/content/data")
    elif os.path.exists(LOCAL_ZIP):
        print(f"Extracting dataset from local runtime: {LOCAL_ZIP}")
        with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
            zip_ref.extractall("/content/data")
    else:
        print("Note: If uploading dataset, please upload 'dataset.zip' to /content/ or mount Google Drive.")

# Verify directory structure
BASE_DIR = Path("/content/data/processed") if os.path.exists("/content/data/processed") else Path("./data/processed")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

print(f"Train directory: {TRAIN_DIR}")
print(f"Val directory:   {VAL_DIR}")
print(f"Test directory:  {TEST_DIR}")""")

    # Cell 5: Dataset verification
    add_code("""# 4. Dataset Validation & Class Mapping (15 Classes)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

CLASS_NAMES = train_ds_raw.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"\\nDiscovered {NUM_CLASSES} Classes:")
for idx, cname in enumerate(CLASS_NAMES):
    print(f"  {idx:2d}: {cname}")

# Save class indices mapping
class_indices = {cname: idx for idx, cname in enumerate(CLASS_NAMES)}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)""")

    # Cell 6: Data Augmentation & Preprocessing
    add_code("""# 5. Data Augmentation & MobileNetV2 Preprocessing Pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=SEED),
    layers.RandomRotation(0.15, seed=SEED),
    layers.RandomZoom(0.15, seed=SEED),
    layers.RandomTranslation(0.1, 0.1, seed=SEED),
], name="data_augmentation")

# Apply augmentation to training set only; apply MobileNetV2 preprocess_input [-1, 1] to all sets
train_ds = train_ds_raw.map(
    lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds_raw.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds_raw.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

print("Data pipelines built and prefetched successfully.")""")

    # Cell 7: Model Architecture
    add_code("""# 6. Build MobileNetV2 Transfer Learning Architecture
def build_mobilenetv2_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    
    # Pretrained MobileNetV2 base (ImageNet weights)
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Freeze base model
    base_model.trainable = False
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CropDisease_MobileNetV2")
    return model, base_model

model, base_model = build_mobilenetv2_model(NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()""")

    # Cell 8: Callbacks
    add_code("""# 7. Setup Training Callbacks
callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath="crop_model.keras",
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
]""")

    # Cell 9: Phase 1 Training
    add_code("""# 8. Phase 1: Feature Extraction (Training Classification Head)
INITIAL_EPOCHS = 8
print(f"Starting Phase 1 Training ({INITIAL_EPOCHS} epochs, base frozen)...")
start_phase1 = time.time()

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks_list
)
print(f"Phase 1 completed in {(time.time() - start_phase1)/60:.2f} minutes.")""")

    # Cell 10: Phase 2 Fine-Tuning
    add_code("""# 9. Phase 2: Fine-Tuning Top Layers
# Unfreeze from layer 120 upwards (MobileNetV2 has 154 layers total)
base_model.trainable = True
for layer in base_model.layers[:120]:
    layer.trainable = False
for layer in base_model.layers[120:]:
    layer.trainable = True

# Recompile with lower learning rate
FINE_TUNE_LR = 1e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_TUNE_EPOCHS = 6
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

print(f"Starting Phase 2 Fine-Tuning ({FINE_TUNE_EPOCHS} additional epochs, lr={FINE_TUNE_LR})...")
start_phase2 = time.time()

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history_phase1.history["loss"]),
    epochs=TOTAL_EPOCHS,
    callbacks=callbacks_list
)
print(f"Phase 2 completed in {(time.time() - start_phase2)/60:.2f} minutes.")""")

    # Cell 11: History plot
    add_code("""# 10. Combine History and Plot Training Curves
acc = history_phase1.history["accuracy"] + history_phase2.history["accuracy"]
val_acc = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
loss = history_phase1.history["loss"] + history_phase2.history["loss"]
val_loss = history_phase1.history["val_loss"] + history_phase2.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, [a * 100 for a in acc], 'g-o', label='Training Accuracy')
plt.plot(epochs_range, [a * 100 for a in val_acc], 'b-s', label='Validation Accuracy')
plt.axvline(x=INITIAL_EPOCHS, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.5)
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'r-o', label='Training Loss')
plt.plot(epochs_range, val_loss, 'm-s', label='Validation Loss')
plt.axvline(x=INITIAL_EPOCHS, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss (Categorical Crossentropy)')
plt.grid(True, alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()""")

    # Cell 12: Comprehensive Evaluation on Test Set
    add_code("""# 11. Rigorous Model Evaluation on Untouched 15% Test Dataset
print("Evaluating best model on untouched 15% test dataset...")
best_model = tf.keras.models.load_model("crop_model.keras")

y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_pred_probs.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.argmax(np.array(y_pred_probs), axis=1)

# Overall Accuracy
test_acc = np.mean(y_true == y_pred) * 100
print(f"\\nFinal Test Accuracy: {test_acc:.2f}%")

# Sklearn Classification Report
report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("classification_report.csv")
print("\\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
plt.ylabel('Actual True Class', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix on Test Dataset (Test Accuracy: {test_acc:.2f}%)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()""")

    # Cell 13: Export Artifacts
    add_code("""# 12. Package and Download Artifacts for Local Deployment
from google.colab import files

artifacts = [
    "crop_model.keras",
    "class_indices.json",
    "classification_report.csv",
    "confusion_matrix.png",
    "training_curves.png"
]

# Create zip archive of trained artifacts
shutil.make_archive("crop_model_artifacts", "zip", ".", ".")
print("Created crop_model_artifacts.zip containing trained weights and evaluation reports.")

# Trigger browser download in Colab
files.download("crop_model_artifacts.zip")
print("Download started! Unpack these files into your local project root: models/ and evaluation/")""")

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Successfully generated Google Colab Notebook at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    create_notebook()
