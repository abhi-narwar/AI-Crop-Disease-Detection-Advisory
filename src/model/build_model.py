"""
MobileNetV2 Transfer Learning Architecture Module
AI-Enabled Crop Disease Detection and Advisory System

Constructs and compiles MobileNetV2 architecture with custom classification head:
- ImageNet pretrained weights.
- MobileNetV2 preprocessing integrated into pipeline.
- GlobalAveragePooling2D + Dropout + Softmax Dense layer.
- Phase 1: Feature extraction (frozen base).
- Phase 2: Fine-tuning capability (unfreezing top N convolutional layers).
"""

import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

def create_crop_disease_model(num_classes: int, learning_rate: float = 1e-3, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Builds the MobileNetV2 transfer learning model.
    Initially freezes all base model layers for feature extraction.
    """
    # 1. Input Layer
    inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")

    # 2. Base Pretrained MobileNetV2 without classification head
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )
    # Freeze base model weights initially
    base_model.trainable = False

    # 3. Connect Base Model
    x = base_model(inputs, training=False)

    # 4. Global Average Pooling & Regularization
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="top_dropout")(x)

    # 5. Classification Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropDisease_MobileNetV2")

    # 6. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model, base_model

def setup_finetuning(model: tf.keras.Model, base_model: tf.keras.Model, unfreeze_from_layer: int = 120, fine_tune_lr: float = 1e-5):
    """
    Unfreezes top layers of MobileNetV2 for second-stage fine-tuning.
    Uses lower learning rate to prevent catastrophic forgetting.
    """
    base_model.trainable = True
    
    # Freeze all layers before unfreeze_from_layer
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from_layer:]:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Fine-tuning configured: Unfroze layers from {unfreeze_from_layer} upwards.")
    print(f"Trainable parameters: {trainable_count:,}")
    return model

if __name__ == "__main__":
    indices_file = MODELS_DIR / "class_indices.json"
    if indices_file.exists():
        with open(indices_file, "r") as f:
            classes = json.load(f)
        num_classes = len(classes)
    else:
        num_classes = 15

    print(f"Building MobileNetV2 architecture for {num_classes} classes...")
    model, base = create_crop_disease_model(num_classes=num_classes)
    model.summary()
