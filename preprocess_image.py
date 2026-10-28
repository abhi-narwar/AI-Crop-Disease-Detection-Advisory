
from pathlib import Path
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    """
    Prepare a single image for MobileNetV2.
    Returns an array with the shape (1, 224, 224, 3).
    """

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(image, dtype=np.float32)

    # MobileNetV2 preprocessing:
    # Convert pixel values from [0, 255] to [-1, 1]
    image_array = image_array / 127.5 - 1.0

    # Add a batch dimension for model input
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python preprocess_image.py '
            '"data/raw/path-to-leaf.jpg"'
        )
        raise SystemExit(1)

    try:
        result = preprocess_image(sys.argv[1])

        print("\nImage preprocessing successful!")
        print("Array shape:", result.shape)
        print("Data type:", result.dtype)
        print("Minimum pixel value:", float(result.min()))
        print("Maximum pixel value:", float(result.max()))

    except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
        print("Error:", error)
        raise SystemExit(1)
