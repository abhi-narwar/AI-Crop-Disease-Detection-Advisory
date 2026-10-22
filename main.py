
from fastapi import FastAPI

app = FastAPI(
    title="AI Crop Disease Detection API",
    description="Backend for the Crop Disease Detection and Advisory System",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "message": "AI Crop Disease Detection System is running!",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": False,
    }


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            "Healthy",
            "Bacterial spot",
            "Early blight",
            "Late blight",
            "Septoria leaf spot",
        ],
        "status": "proposed_classes"
    }
