"""
Automated Integration Tests for FastAPI Endpoints
"""

import os
import sys
import io
from pathlib import Path
import pytest

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure venv packages are discoverable before importing fastapi
venv_site = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

from fastapi import UploadFile, HTTPException
from backend.routes.health import health_check
from backend.routes.classes import get_supported_classes
from backend.routes.predict import predict_crop_disease

def test_api_health_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_status"]["loaded"] is True
    assert res["model_status"]["classes_count"] == 15

def test_api_classes_endpoint():
    res = get_supported_classes()
    assert res["status"] == "success"
    assert res["total_classes"] == 15
    assert set(res["crops"]) == {"Pepper (Bell Pepper)", "Potato", "Tomato"}

def test_api_predict_real_image():
    import pandas as pd
    manifest = PROJECT_ROOT / "data" / "manifests" / "test.csv"
    df = pd.read_csv(manifest)
    sample_path = df.iloc[0]["filepath"]

    with open(sample_path, "rb") as f:
        data = f.read()

    file_obj = UploadFile(
        file=io.BytesIO(data),
        filename="test_leaf.jpg"
    )

    res = predict_crop_disease(file_obj)
    assert res["status"] == "success"
    assert res["predicted_crop"] in ["Pepper (Bell Pepper)", "Potato", "Tomato"]
    assert res["confidence"] > 0
    assert len(res["top_predictions"]) == 3
    assert "disclaimer" in res["advisory"]

def test_api_predict_empty_file():
    file_obj = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.jpg"
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_crop_disease(file_obj)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()

def test_api_predict_invalid_extension():
    file_obj = UploadFile(
        file=io.BytesIO(b"dummy pdf content"),
        filename="document.pdf"
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_crop_disease(file_obj)
    assert exc_info.value.status_code == 400
    assert "unsupported" in exc_info.value.detail.lower()

def test_api_predict_corrupted_image():
    file_obj = UploadFile(
        file=io.BytesIO(b"not a valid jpeg or png file data"),
        filename="corrupted.jpg"
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_crop_disease(file_obj)
    assert exc_info.value.status_code == 400
    assert "decoded" in exc_info.value.detail.lower()
