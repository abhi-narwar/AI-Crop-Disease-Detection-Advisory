"""
Prediction Endpoint
POST /predict
Accepts leaf image upload, validates, infers disease condition, and returns advisory.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from backend.services.prediction_service import prediction_service

router = APIRouter(tags=["Inference & Advisory"])

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

@router.post("/predict", summary="Predict Crop Disease and Return Agricultural Advisory")
def predict_crop_disease(file: UploadFile = File(..., description="Crop leaf image file")):
    # 1. Check filename and extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload."
        )

    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Read bytes and check size
    if hasattr(file, "file") and hasattr(file.file, "read"):
        contents = file.file.read()
    elif hasattr(file, "read"):
        contents = file.read()
    else:
        contents = b""
    file_size = len(contents)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes). Please upload a valid image."
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({round(file_size / (1024*1024), 2)} MB) exceeds maximum allowed limit of 10 MB."
        )

    # 3. Check model readiness
    model_status = prediction_service.get_model_status()
    if not model_status["model_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The disease detection model is currently not loaded. Please ensure model training is complete."
        )

    # 4. Validate image integrity
    try:
        import io
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The uploaded file could not be decoded as a valid image: {str(e)}"
        )

    # 5. Run inference
    try:
        result = prediction_service.predict_image(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed during model inference: {str(e)}"
        )

    return {
        "status": "success",
        "filename": file.filename,
        "file_size_kb": round(file_size / 1024, 2),
        "predicted_crop": result["predicted_crop"],
        "predicted_disease": result["predicted_disease"],
        "is_healthy": result["is_healthy"],
        "confidence": result["confidence"],
        "top_predictions": result["top_predictions"],
        "advisory": result["advisory"]
    }
