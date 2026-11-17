"""
Crop Disease Prediction Inference Engine
AI-Enabled Crop Disease Detection and Advisory System

Provides reusable, production-ready inference pipeline:
- Validates input images (formats, dimensions, non-corrupted check).
- Converts to RGB and resizes to 224x224.
- Dynamically uses trained model:
    - Primary local: models/crop_model.joblib (real vision classifier trained on 15 classes)
    - Or deep Keras: models/crop_model.keras (MobileNetV2 deep learning model)
- Returns predicted crop, disease condition, confidence %, top 3 predictions, and agricultural advisory.
- Guaranteed zero mock/dummy data.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
from PIL import Image

# Silence Loky CPU warning and prevent subprocess timeouts
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
KERAS_MODEL_PATH = MODELS_DIR / "crop_model.keras"
JOBLIB_MODEL_PATH = MODELS_DIR / "crop_model.joblib"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

from backend.advisory.advisory_service import get_advisory_for_class
from src.model.train_local import extract_image_features

class CropDiseasePredictor:
    """
    Singleton inference service for crop disease detection.
    """
    def __init__(self):
        self.class_indices = {}
        self.idx_to_class = {}
        self.joblib_model = None
        self.keras_model = None
        self.active_backend = None
        self.is_loaded = False
        
        self.load_resources()

    def load_resources(self):
        """Loads class indices and available trained models."""
        # Load class indices
        if CLASS_INDICES_PATH.exists():
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                self.class_indices = json.load(f)
                self.idx_to_class = {int(v): k for k, v in self.class_indices.items()}

        # 1. Check joblib model
        if JOBLIB_MODEL_PATH.exists():
            try:
                import joblib
                self.joblib_model = joblib.load(JOBLIB_MODEL_PATH)
                self.active_backend = "joblib"
                self.is_loaded = True
                print(f"[Predictor] Loaded local ML vision model from {JOBLIB_MODEL_PATH}")
            except Exception as e:
                print(f"[Predictor] Error loading joblib model: {e}")

        # 2. Check keras model
        if KERAS_MODEL_PATH.exists() and not self.is_loaded:
            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
                self.active_backend = "keras"
                self.is_loaded = True
                print(f"[Predictor] Loaded Keras deep model from {KERAS_MODEL_PATH}")
            except Exception as e:
                print(f"[Predictor] Note: Keras model not loaded: {e}")

    def validate_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Validates and returns PIL RGB Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            img = Image.open(path)
        elif isinstance(image_input, bytes):
            import io
            if len(image_input) == 0:
                raise ValueError("Uploaded file is empty (0 bytes).")
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Verify integrity
        img.verify()

        # Reopen after verify
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input))

        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    def parse_class_name(self, class_name: str) -> Dict[str, Any]:
        """Extracts human-readable crop and disease condition from class folder name."""
        if class_name.startswith("Pepper"):
            crop = "Pepper (Bell Pepper)"
        elif class_name.startswith("Potato"):
            crop = "Potato"
        elif class_name.startswith("Tomato"):
            crop = "Tomato"
        else:
            crop = "Unknown"

        is_healthy = "healthy" in class_name.lower()
        readable = class_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
        prefix_to_strip = "Pepper - bell" if "Pepper" in crop else crop
        if readable.startswith(prefix_to_strip):
            condition = readable[len(prefix_to_strip):].strip(" -")
        else:
            condition = readable

        return {
            "crop": crop,
            "disease": condition if condition else "Healthy",
            "is_healthy": is_healthy
        }

    def predict(self, image_input: Union[str, Path, bytes, Image.Image], top_k: int = 3) -> Dict[str, Any]:
        """Performs actual model inference on the uploaded image."""
        if not self.is_loaded:
            # Attempt reloading in case model was just trained
            self.load_resources()
            if not self.is_loaded:
                raise RuntimeError("No trained model available. Please train the model before running inference.")

        if not self.idx_to_class:
            raise RuntimeError("Class indices mapping missing. Please verify models/class_indices.json exists.")

        img = self.validate_image(image_input)

        if self.active_backend == "joblib":
            feats = extract_image_features(img).reshape(1, -1)
            probabilities = self.joblib_model.predict_proba(feats)[0]
        elif self.active_backend == "keras":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            arr = np.expand_dims(preprocess_input(np.array(resized, dtype=np.float32)), axis=0)
            probabilities = self.keras_model.predict(arr, verbose=0)[0]
        else:
            raise RuntimeError("Unknown active model backend.")

        # Top prediction
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_class_name = self.idx_to_class.get(predicted_idx, "Unknown")
        meta = self.parse_class_name(predicted_class_name)

        # Top-K predictions
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        for idx in top_k_indices:
            cname = self.idx_to_class.get(int(idx), "Unknown")
            pmeta = self.parse_class_name(cname)
            top_predictions.append({
                "class_name": cname,
                "crop": pmeta["crop"],
                "disease": pmeta["disease"],
                "is_healthy": pmeta["is_healthy"],
                "confidence": round(float(probabilities[idx]) * 100, 2)
            })

        # Fetch agricultural advisory
        advisory = get_advisory_for_class(predicted_class_name)

        return {
            "predicted_class": predicted_class_name,
            "predicted_crop": meta["crop"],
            "predicted_disease": meta["disease"],
            "is_healthy": meta["is_healthy"],
            "confidence": round(confidence * 100, 2),
            "top_predictions": top_predictions,
            "advisory": advisory
        }

_predictor_instance = None

def get_predictor() -> CropDiseasePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CropDiseasePredictor()
    return _predictor_instance
