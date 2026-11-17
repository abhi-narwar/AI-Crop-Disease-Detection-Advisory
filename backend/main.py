"""
FastAPI Main Application
AI-Enabled Crop Disease Detection and Advisory System

Runs the REST API backend serving:
- GET /health
- GET /classes
- POST /predict
- Mounts and serves frontend UI
- Swagger UI at /docs
"""

import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes.health import router as health_router
from backend.routes.classes import router as classes_router
from backend.routes.predict import router as predict_router

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="AI-Enabled Crop Disease Detection and Advisory System API",
    description="Deep Learning REST API for plant leaf disease classification across 15 agricultural crop classes with agronomic advisory.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health_router)
app.include_router(classes_router)
app.include_router(predict_router)

# Mount frontend assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse, tags=["Frontend UI"])
def serve_home():
    """Serves the frontend single-page application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>AI Crop Disease Detection API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger documentation.</p>", status_code=200)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected internal server error occurred.",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main.py:app", host="127.0.0.1", port=8000, reload=True)
