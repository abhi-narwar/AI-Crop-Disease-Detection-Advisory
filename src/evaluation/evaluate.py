"""
Model Evaluation Module
AI-Enabled Crop Disease Detection and Advisory System

Evaluates trained model on the untouched 15% test dataset (3,091 images).
Calculates:
- Overall Accuracy
- Precision, Recall, F1-Score (Macro and Weighted averages)
- Per-class classification report
- High-resolution Confusion Matrix heatmap
- Sample prediction visualizations

Saves:
- evaluation/classification_report.txt
- evaluation/classification_report.csv
- evaluation/confusion_matrix.png
- evaluation/evaluation_summary.json
- evaluation/sample_predictions.png
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import joblib

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib setup
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.train_local import extract_image_features

def evaluate_model():
    print("=" * 70)
    print("AI-ENABLED CROP DISEASE DETECTION: TEST SET EVALUATION")
    print("=" * 70)

    # 1. Load test manifest
    test_manifest_path = MANIFESTS_DIR / "test.csv"
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest_path}")

    df_test = pd.read_csv(test_manifest_path)
    print(f"Loaded test dataset: {len(df_test):,} images across {df_test['class_name'].nunique()} classes.")

    # 2. Load class indices
    indices_path = MODELS_DIR / "class_indices.json"
    with open(indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(class_indices))]

    # 3. Load trained model
    model_path = MODELS_DIR / "crop_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")

    clf = joblib.load(model_path)
    print(f"Loaded trained model from {model_path}")

    # 4. Extract features on untouched test images
    print("\nExtracting feature vectors for untouched test dataset...")
    X_test = []
    y_true = []
    filepaths = []
    
    for idx, row in df_test.iterrows():
        try:
            feats = extract_image_features(row["filepath"])
            X_test.append(feats)
            y_true.append(int(row["class_index"]))
            filepaths.append(row["filepath"])
        except Exception as e:
            continue

    X_test = np.array(X_test, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)
    print(f"Extracted {X_test.shape[0]} test feature vectors of shape {X_test.shape[1]}.")

    # 5. Run actual inference
    print("\nExecuting model predictions on test dataset...")
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)

    # 6. Calculate Metrics
    test_accuracy = accuracy_score(y_true, y_pred) * 100
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # 7. Classification Report (Text & CSV)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    with open(EVALUATION_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(EVALUATION_DIR / "classification_report.csv")

    # 8. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Ground Truth Label", fontsize=12, fontweight="bold")
    plt.title(f"Confusion Matrix on Held-Out Test Set (Test Accuracy: {test_accuracy:.2f}%)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    # 9. Generate Sample Predictions Grid
    generate_sample_predictions_grid(filepaths, y_true, y_pred, y_probs, idx_to_class)

    # 10. Save Evaluation Summary JSON
    summary = {
        "dataset": "PlantVillage Test Split (Untouched 15%)",
        "total_test_samples": len(y_true),
        "num_classes": len(class_names),
        "test_accuracy": round(float(test_accuracy), 2),
        "macro_avg": {
            "precision": round(float(precision_macro) * 100, 2),
            "recall": round(float(recall_macro) * 100, 2),
            "f1_score": round(float(f1_macro) * 100, 2)
        },
        "weighted_avg": {
            "precision": round(float(precision_weighted) * 100, 2),
            "recall": round(float(recall_weighted) * 100, 2),
            "f1_score": round(float(f1_weighted) * 100, 2)
        },
        "per_class_f1": {cname: round(float(report_dict[cname]["f1-score"]) * 100, 2) for cname in class_names},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(EVALUATION_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS:")
    print("=" * 70)
    print(f"Test Accuracy:     {test_accuracy:.2f}%")
    print(f"Macro F1-Score:    {summary['macro_avg']['f1_score']}%")
    print(f"Weighted F1-Score: {summary['weighted_avg']['f1_score']}%")
    print("\nClassification Report Preview:\n")
    print(report_text)
    print("=" * 70)

    return summary

def generate_sample_predictions_grid(filepaths, y_true, y_pred, y_probs, idx_to_class, num_samples=6):
    """Visualizes sample test images with actual vs predicted labels and confidence."""
    indices = np.random.RandomState(42).choice(len(filepaths), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_path = filepaths[idx]
        true_label = idx_to_class[y_true[idx]].replace("___", "\n").replace("__", "\n")
        pred_label = idx_to_class[y_pred[idx]].replace("___", "\n").replace("__", "\n")
        conf = float(y_probs[idx][y_pred[idx]]) * 100
        is_correct = y_true[idx] == y_pred[idx]

        with Image.open(img_path) as img:
            axes[i].imshow(img)
            axes[i].axis("off")
            title_color = "green" if is_correct else "red"
            status = "CORRECT" if is_correct else "MISCLASSIFIED"
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.1f}% [{status}]",
                              fontsize=9, fontweight="bold", color=title_color)

    plt.tight_layout()
    sample_path = EVALUATION_DIR / "sample_predictions.png"
    plt.savefig(sample_path, dpi=300)
    plt.close()
    print(f"Saved sample prediction visualizations to {sample_path}")

if __name__ == "__main__":
    evaluate_model()
