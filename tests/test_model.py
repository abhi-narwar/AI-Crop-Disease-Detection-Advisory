"""
Automated Unit Tests for Model Inference Engine
"""

import sys
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict import get_predictor

def test_model_loaded():
    predictor = get_predictor()
    assert predictor.is_loaded is True, "Predictor model must be loaded successfully"
    assert len(predictor.class_indices) == 15, "Predictor must have 15 classes"

def test_prediction_on_real_image():
    predictor = get_predictor()
    
    # Load first image from test manifest
    import pandas as pd
    df_test = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "test.csv")
    sample_path = df_test.iloc[0]["filepath"]

    result = predictor.predict(sample_path)
    assert "predicted_class" in result
    assert "predicted_crop" in result
    assert "predicted_disease" in result
    assert "confidence" in result
    assert "top_predictions" in result
    assert "advisory" in result

    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 100.0
    assert len(result["top_predictions"]) == 3
    assert result["top_predictions"][0]["confidence"] >= result["top_predictions"][1]["confidence"]
    assert result["top_predictions"][1]["confidence"] >= result["top_predictions"][2]["confidence"]

def test_invalid_image_handling():
    predictor = get_predictor()
    
    with pytest.raises(Exception):
        # Empty byte string should raise exception
        predictor.predict(b"")

    with pytest.raises(Exception):
        # Non-image data
        predictor.predict(b"not an image file content")
