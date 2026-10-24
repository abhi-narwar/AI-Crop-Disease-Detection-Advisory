
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="AI Crop Disease Detection API",
    description="Crop leaf image upload and validation",
    version="0.2.0",
)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


from fastapi.responses import RedirectResponse

@app.get("/")
def home():
    return RedirectResponse(url="/docs")


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
        "status": "proposed_classes",
    }


@app.post("/upload")
async def upload_leaf(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    # 1. Check file extension
    extension = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed.",
        )

    try:
        # 2. Read at most 5 MB plus one byte
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 5 MB.",
            )

        # 3. Verify that the file contains a real image
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400,
                    detail="File content must be a valid JPEG or PNG image.",
                )

            image.verify()

        # 4. Return basic information; no prediction yet
        return {
            "message": "Image uploaded and validated successfully.",
            "filename": file.filename,
            "format": extension.lstrip(".").upper(),
            "size_bytes": len(content),
            "prediction": None,
            "note": "AI model integration will be added later.",
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a readable image.",
        ) from None

    except (OSError, SyntaxError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The image is corrupt or cannot be read.",
        ) from None

    finally:
        await file.close()
