"""
Classes Information Endpoint
GET /classes
"""

import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(tags=["Dataset & Classes"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES_PATH = PROJECT_ROOT / "models" / "class_indices.json"

@router.get("/classes", summary="Get Supported Crops and Disease Classes")
def get_supported_classes():
    if not CLASSES_PATH.exists():
        return {
            "status": "error",
            "message": "Class indices file not found."
        }

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    crops_structure = {
        "Pepper (Bell Pepper)": [],
        "Potato": [],
        "Tomato": []
    }

    all_classes_list = []

    for cname, idx in class_indices.items():
        if cname.startswith("Pepper"):
            crop = "Pepper (Bell Pepper)"
        elif cname.startswith("Potato"):
            crop = "Potato"
        else:
            crop = "Tomato"

        is_healthy = "healthy" in cname.lower()
        readable = cname.replace("___", " - ").replace("__", " - ").replace("_", " ")
        prefix_to_strip = "Pepper - bell" if "Pepper" in crop else crop
        condition = readable[len(prefix_to_strip):].strip(" -") if readable.startswith(prefix_to_strip) else readable
        if not condition:
            condition = "Healthy"

        item = {
            "class_index": idx,
            "raw_class_name": cname,
            "crop": crop,
            "condition": condition,
            "is_healthy": is_healthy
        }

        crops_structure[crop].append(item)
        all_classes_list.append(item)

    return {
        "status": "success",
        "total_classes": len(all_classes_list),
        "crops": list(crops_structure.keys()),
        "classes_by_crop": crops_structure,
        "classes": all_classes_list
    }
