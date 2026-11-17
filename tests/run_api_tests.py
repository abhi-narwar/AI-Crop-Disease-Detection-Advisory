"""
Standalone Asynchronous API Test Runner
Tests FastAPI endpoints in-process via httpx.ASGITransport without socket binding.
"""

import os
import sys
import io
import asyncio
from pathlib import Path
import pandas as pd
from PIL import Image
import httpx

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

async def main():
    print("=" * 70)
    print("FASTAPI BACKEND ASYNCHRONOUS INTEGRATION TESTS")
    print("=" * 70)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Test /health
        print("\n1. Testing GET /health...")
        r = await client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        health_data = r.json()
        print(f"   Status: {health_data['status']} | Model Loaded: {health_data['model_status']['loaded']}")
        assert health_data["status"] == "healthy"
        assert health_data["model_status"]["classes_count"] == 15
        print("   ✓ Health check passed!")

        # 2. Test /classes
        print("\n2. Testing GET /classes...")
        r = await client.get("/classes")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        classes_data = r.json()
        print(f"   Total classes tracked: {classes_data['total_classes']}")
        assert classes_data["total_classes"] == 15
        assert len(classes_data["crops"]) == 3
        print("   ✓ Classes catalog endpoint passed!")

        # 3. Test /predict with a real held-out test leaf image
        print("\n3. Testing POST /predict with real test image...")
        test_manifest = PROJECT_ROOT / "data" / "manifests" / "test.csv"
        df_test = pd.read_csv(test_manifest)
        real_sample_path = df_test.iloc[0]["filepath"]
        print(f"   Uploading real test image: {real_sample_path}")

        with open(real_sample_path, "rb") as f:
            img_bytes = f.read()

        r = await client.post(
            "/predict",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        pred_data = r.json()
        print(f"   Predicted Crop:    {pred_data['predicted_crop']}")
        print(f"   Predicted Disease: {pred_data['predicted_disease']}")
        print(f"   Condition:         {'Healthy' if pred_data['is_healthy'] else 'Diseased'}")
        print(f"   Confidence:        {pred_data['confidence']}%")
        print(f"   Top 3 Count:       {len(pred_data['top_predictions'])}")
        print(f"   Advisory Included: Symptoms: {len(pred_data['advisory']['symptoms'])}, Prevention: {len(pred_data['advisory']['prevention'])}")
        assert pred_data["status"] == "success"
        assert pred_data["confidence"] > 0
        assert len(pred_data["top_predictions"]) == 3
        assert "disclaimer" in pred_data["advisory"]
        print("   ✓ Real image prediction and advisory passed!")

        # 4. Test Error Handling: Empty Upload
        print("\n4. Testing POST /predict with empty file (0 bytes)...")
        r = await client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        print(f"   Status Code: {r.status_code} | Error: {r.json()['detail']}")
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()
        print("   ✓ Empty upload rejection passed!")

        # 5. Test Error Handling: Invalid File Extension
        print("\n5. Testing POST /predict with invalid file type (PDF)...")
        r = await client.post(
            "/predict",
            files={"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")}
        )
        print(f"   Status Code: {r.status_code} | Error: {r.json()['detail']}")
        assert r.status_code == 400
        assert "unsupported" in r.json()["detail"].lower()
        print("   ✓ Invalid extension rejection passed!")

        # 6. Test Error Handling: Corrupted Image Bytes
        print("\n6. Testing POST /predict with corrupted image...")
        r = await client.post(
            "/predict",
            files={"file": ("corrupted.jpg", b"corrupted random data not an image", "image/jpeg")}
        )
        print(f"   Status Code: {r.status_code} | Error: {r.json()['detail']}")
        assert r.status_code == 400
        assert "decoded" in r.json()["detail"].lower()
        print("   ✓ Corrupted image rejection passed!")

    print("\n" + "=" * 70)
    print("ALL 6 FASTAPI INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
