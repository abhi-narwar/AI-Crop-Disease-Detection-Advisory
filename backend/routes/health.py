"""
Health Check Endpoint
GET /health
"""

from fastapi import APIRouter
from backend.services.prediction_service import prediction_service

router = APIRouter(tags=["System Health"])

@router.get("/health", summary="System and Model Health Check")
def health_check():
    status_info = prediction_service.get_model_status()
    is_ready = status_info["model_loaded"]

    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "AI-Enabled Crop Disease Detection API",
        "version": "1.0.0",
        "model_status": {
            "loaded": status_info["model_loaded"],
            "backend": status_info["active_backend"],
            "classes_count": status_info["classes_registered"],
            "model_version": status_info["model_version"]
        }
    }
