"""
Direct FastAPI Route Unit Tests
Tests route handlers, validation logic, HTTP exception handling, and response payloads directly.
"""

import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

venv_site = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

from fastapi import UploadFile, HTTPException
from backend.routes.health import health_check
from backend.routes.classes import get_supported_classes
from backend.routes.predict import predict_crop_disease

def test_route_health():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["model_status"]["loaded"] is True
    assert res["model_status"]["classes_count"] == 15
    print("✓ Route health_check() passed!")

def test_route_classes():
    res = get_supported_classes()
    assert res["status"] == "success"
    assert res["total_classes"] == 15
    assert len(res["crops"]) == 3
    print("✓ Route get_supported_classes() passed!")

def test_route_predict_real_image():
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
    print(f"✓ Route predict_crop_disease() passed on real leaf! Result: {res['predicted_crop']} - {res['predicted_disease']} ({res['confidence']}%)")

def test_route_predict_empty_file():
    file_obj = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.jpg"
    )
    caught = False
    try:
        predict_crop_disease(file_obj)
    except HTTPException as e:
        caught = True
        assert e.status_code == 400
        assert "empty" in e.detail.lower()
    assert caught, "Expected HTTPException for empty file"
    print("✓ Route empty file rejection passed (HTTP 400)!")

def test_route_predict_invalid_extension():
    file_obj = UploadFile(
        file=io.BytesIO(b"dummy pdf content"),
        filename="document.pdf"
    )
    caught = False
    try:
        predict_crop_disease(file_obj)
    except HTTPException as e:
        caught = True
        assert e.status_code == 400
        assert "unsupported" in e.detail.lower()
    assert caught, "Expected HTTPException for invalid extension"
    print("✓ Route invalid extension rejection passed (HTTP 400)!")

def test_route_predict_corrupt_image():
    file_obj = UploadFile(
        file=io.BytesIO(b"not a valid jpeg or png file data"),
        filename="corrupted.jpg"
    )
    caught = False
    try:
        predict_crop_disease(file_obj)
    except HTTPException as e:
        caught = True
        assert e.status_code == 400
        assert "decoded" in e.detail.lower()
    assert caught, "Expected HTTPException for corrupted image"
    print("✓ Route corrupted image rejection passed (HTTP 400)!")

def test_route_predict_oversized_file():
    huge_data = b"0" * (11 * 1024 * 1024)
    file_obj = UploadFile(
        file=io.BytesIO(huge_data),
        filename="huge.jpg"
    )
    caught = False
    try:
        predict_crop_disease(file_obj)
    except HTTPException as e:
        caught = True
        assert e.status_code == 413
        assert "exceeds" in e.detail.lower()
    assert caught, "Expected HTTPException for oversized file"
    print("✓ Route oversized file rejection passed (HTTP 413)!")

if __name__ == "__main__":
    print("=" * 70)
    print("FASTAPI DIRECT ROUTE & EXCEPTION TEST SUITE")
    print("=" * 70)
    test_route_health()
    test_route_classes()
    test_route_predict_real_image()
    test_route_predict_empty_file()
    test_route_predict_invalid_extension()
    test_route_predict_corrupt_image()
    test_route_predict_oversized_file()
    print("\n" + "=" * 70)
    print("ALL 7 DIRECT FASTAPI ROUTE & VALIDATION TESTS PASSED!")
    print("=" * 70)
