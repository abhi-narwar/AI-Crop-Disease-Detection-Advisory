"""
Comprehensive Per-Class Integration Test
Tests an actual held-out test set image from each of the 15 supported classes.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict import get_predictor

def test_prediction_coverage_all_15_classes():
    predictor = get_predictor()
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "test.csv"
    assert manifest_path.exists(), "test.csv must exist"

    df_test = pd.read_csv(manifest_path)
    unique_classes = df_test["class_name"].unique()
    assert len(unique_classes) == 15, f"Expected 15 classes in test set, got {len(unique_classes)}"

    print("\nTesting one sample image from each of the 15 classes:")
    class_results = []

    for cname in sorted(unique_classes):
        sample_row = df_test[df_test["class_name"] == cname].iloc[0]
        img_path = sample_row["filepath"]
        
        result = predictor.predict(img_path)
        is_exact_match = result["predicted_class"] == cname

        class_results.append({
            "true_class": cname,
            "pred_class": result["predicted_class"],
            "pred_crop": result["predicted_crop"],
            "confidence": result["confidence"],
            "match": is_exact_match
        })
        
        print(f"  [{cname:40s}] -> Pred: {result['predicted_class']:40s} | Conf: {result['confidence']:5.1f}% | Match: {is_exact_match}")

        # Assert mandatory contract fields
        assert result["predicted_crop"] in ["Pepper (Bell Pepper)", "Potato", "Tomato"]
        assert len(result["top_predictions"]) == 3
        assert "symptoms" in result["advisory"]
        assert "prevention" in result["advisory"]
        assert "management" in result["advisory"]
        assert "disclaimer" in result["advisory"]

    # Verify that at least 70% of sample images from test set match their ground truth
    matches = sum(1 for r in class_results if r["match"])
    match_rate = matches / len(class_results) * 100
    print(f"\nTested 15 classes: {matches}/15 exact top-1 matches ({match_rate:.1f}% sample accuracy)")
    assert matches >= 10, f"Expected at least 10/15 top-1 matches, got {matches}"

if __name__ == "__main__":
    test_prediction_coverage_all_15_classes()
