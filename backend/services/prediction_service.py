"""
Prediction Service
AI-Enabled Crop Disease Detection and Advisory System

Encapsulates prediction logic and model interaction for FastAPI endpoints.
"""

from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image

from src.model.predict import get_predictor

class PredictionService:
    def __init__(self):
        self.predictor = get_predictor()

    def get_model_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.predictor.is_loaded,
            "active_backend": self.predictor.active_backend or "none",
            "classes_registered": len(self.predictor.class_indices),
            "model_version": "1.0.0"
        }

    def predict_image(self, image_data: Union[bytes, Image.Image, Path, str]) -> Dict[str, Any]:
        return self.predictor.predict(image_data)

prediction_service = PredictionService()
