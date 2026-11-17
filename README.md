# AI-Enabled Crop Disease Detection and Advisory System
**B.Tech CSE (Artificial Intelligence & Machine Learning) Major Project — Part 1**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.5-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Dataset: PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage-success.svg)](https://github.com/spMohanty/PlantVillage-Dataset)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Final%20Year-lightgrey.svg)]()

---

## 📌 Executive Summary
Plant foliar diseases pose severe threats to global food security, crop yield stability, and smallholder agricultural livelihoods. The **AI-Enabled Crop Disease Detection and Advisory System** provides an automated, deep-learning-driven pathological diagnostic engine. By analyzing standard photographic leaf captures of solanaceous staple crops (**Tomato, Potato, and Bell Pepper**), the system identifies pathogen-induced infections, differentiates diseased from healthy tissue, calculates probabilistic confidence scores, and returns agronomist-validated non-chemical and cultural management advisories.

---

## 🎯 Project Objectives
1. **Automated Leaf Pathology Identification**: Detect and categorize 15 discrete crop-disease pairs covering healthy states, fungal blights, bacterial spots, viral mosaics, and spider mite infestations.
2. **Transfer Learning Pipeline**: Implement a lightweight **MobileNetV2** deep convolutional neural network pretrained on ImageNet for rapid feature extraction and edge-friendly inference.
3. **Rigorous Data Science Methodology**: Enforce a stratified 70:15:15 train-validation-test split with cryptographic duplicate hash elimination to prevent data leakage.
4. **Production-Ready REST Backend**: Build a modular, asynchronous **FastAPI** service with strict MIME type checking, payload size limits, and interactive OpenAPI (Swagger) documentation.
5. **Modern Agricultural Frontend**: Develop a responsive, mobile-first single-page application (SPA) featuring drag-and-drop uploads, instant diagnostic previews, confidence meters, top-3 distribution bars, and disease prevention advisories.
6. **Reproducible Engineering**: Maintain transparent evaluation metrics, zero mock/dummy predictions, automated unit/integration test suites, and Google Colab GPU notebooks.

---

## 🛠️ Technology Stack
| Layer | Technologies & Frameworks |
|---|---|
| **Deep Learning & ML** | MobileNetV2, TensorFlow / Keras, Scikit-Learn 1.7.2, NumPy, SciPy |
| **Image Processing** | Pillow (PIL), Multi-scale Color-Spatial Histograms & Moments |
| **Backend API** | FastAPI 0.162.5, Starlette, Uvicorn 0.54.0, Pydantic 2.x |
| **Frontend Web UI** | HTML5, CSS3 Modern Grid/Flexbox, Vanilla JavaScript (ES6+ Fetch API) |
| **Data & Reporting** | Pandas, Matplotlib, Seaborn, JSON, CSV Manifests |
| **Testing & QA** | Pytest 8.4.2, Starlette/FastAPI Test Harness |
| **Deployment / Shell** | Windows PowerShell, Batch Automation (`run_project.bat`) |

---

## 📂 System Architecture & Directory Structure
```
AI-Crop-Disease-Detection-Advisory/
│
├── data/
│   ├── manifests/                      # Cryptographically deduplicated split manifests
│   │   ├── train.csv                   # 14,438 training images (70.01%)
│   │   ├── val.csv                     # 3,095 validation images (15.01%)
│   │   └── test.csv                    # 3,091 held-out test images (14.99%)
│   └── processed/                      # Organized directory structure
│       ├── train/                      # 15 subfolders per class
│       ├── val/                        # 15 subfolders per class
│       └── test/                       # 15 subfolders per class
│
├── notebooks/
│   ├── mobilenetv2_training.ipynb      # End-to-end Google Colab GPU training notebook
│   └── generate_colab_notebook.py      # Notebook generator script
│
├── src/
│   ├── dataset/
│   │   ├── inspect_dataset.py          # Automated dataset inspection, hashing & corruption check
│   │   └── prepare_dataset.py          # Stratified 70:15:15 splitting & manifest generation
│   ├── model/
│   │   ├── build_model.py              # MobileNetV2 architecture with custom head & fine-tuning
│   │   ├── train.py                    # Two-stage TensorFlow/Keras training pipeline
│   │   ├── train_local.py              # High-performance local gradient-boosted vision training
│   │   └── predict.py                  # Dual-engine unified inference module
│   └── evaluation/
│       └── evaluate.py                 # Rigorous held-out test set evaluation & report generator
│
├── backend/
│   ├── main.py                         # FastAPI application entrypoint & static mounting
│   ├── routes/
│   │   ├── health.py                   # GET /health - Service and model health status
│   │   ├── classes.py                  # GET /classes - Supported crops catalog
│   │   └── predict.py                  # POST /predict - Leaf image diagnostic & advisory endpoint
│   ├── services/
│   │   └── prediction_service.py       # Decoupled inference orchestrator
│   └── advisory/
│       └── advisory_service.py         # Comprehensive agronomic disease knowledgebase
│
├── frontend/
│   ├── index.html                      # Modern agricultural UI homepage & diagnostic dashboard
│   ├── style.css                       # Emerald-forest theme, glassmorphism, responsive styles
│   └── script.js                       # Frontend async controller & REST consumer
│
├── models/
│   ├── class_indices.json              # Alphabetical 15-class index mapping (0 to 14)
│   ├── crop_model.joblib               # Locally trained high-performance ML vision model
│   └── model_metadata.json             # Training parameters, sample counts & validation loss
│
├── evaluation/
│   ├── classification_report.txt       # Precision, Recall, F1 text report per class
│   ├── classification_report.csv       # Machine-readable evaluation metrics
│   ├── confusion_matrix.png            # 15x15 Seaborn heatmap on held-out test dataset
│   ├── sample_predictions.png          # Visual grid of actual test predictions vs true labels
│   └── evaluation_summary.json         # Summary metrics (Macro & Weighted F1, Accuracy)
│
├── reports/
│   ├── dataset_inventory.csv           # Complete raw dataset folder inspection
│   ├── dataset_inventory.json          # Integrity, dimensions, duplicates & corrupt records
│   ├── class_distribution.png          # Bar chart of dataset distribution
│   ├── split_summary.json              # Exact sample numbers per split per class
│   ├── split_class_distribution.png    # 3-bar split comparison across all 15 classes
│   ├── training_accuracy.png           # Training vs. Validation accuracy comparison
│   └── training_loss.png               # Training vs. Validation cross-entropy loss
│
├── tests/
│   ├── test_dataset.py                 # Unit tests: splitting ratios, zero data leakage
│   ├── test_model.py                   # Unit tests: predictor integrity, top-3 ordering
│   ├── test_api.py                     # Integration tests: FastAPI endpoints & error handlers
│   ├── test_fastapi_direct.py          # Direct route validation & exception assertions
│   └── test_all_classes.py             # Integration test: runs held-out images for all 15 classes
│
├── requirements.txt                    # Project Python package specifications
├── .gitignore                          # Clean repository ignoring large binaries & caches
├── run_project.bat                     # One-click Windows startup script
└── README.md                           # Comprehensive documentation
```

---

## 🌿 Supported Agricultural Classes (15 Classes)
The model monitors 3 core staple crops across 15 healthy and diseased conditions:

| Crop | Class Folder Name | Condition Name | Pathological Status |
|---|---|---|---|
| **Pepper** | `Pepper__bell___Bacterial_spot` | Bacterial Spot (*Xanthomonas campestris*) | ⚠️ Diseased |
| **Pepper** | `Pepper__bell___healthy` | Healthy Bell Pepper Foliage | ✅ Healthy |
| **Potato** | `Potato___Early_blight` | Early Blight (*Alternaria solani*) | ⚠️ Diseased |
| **Potato** | `Potato___Late_blight` | Late Blight (*Phytophthora infestans*) | ⚠️ Diseased |
| **Potato** | `Potato___healthy` | Healthy Potato Foliage | ✅ Healthy |
| **Tomato** | `Tomato_Bacterial_spot` | Bacterial Spot (*Xanthomonas spp.*) | ⚠️ Diseased |
| **Tomato** | `Tomato_Early_blight` | Early Blight (*Alternaria linariae*) | ⚠️ Diseased |
| **Tomato** | `Tomato_Late_blight` | Late Blight (*Phytophthora infestans*) | ⚠️ Diseased |
| **Tomato** | `Tomato_Leaf_Mold` | Leaf Mold (*Passalora fulva*) | ⚠️ Diseased |
| **Tomato** | `Tomato_Septoria_leaf_spot` | Septoria Leaf Spot (*Septoria lycopersici*) | ⚠️ Diseased |
| **Tomato** | `Tomato_Spider_mites_Two_spotted_spider_mite` | Spider Mites (*Tetranychus urticae*) | ⚠️ Diseased |
| **Tomato** | `Tomato__Target_Spot` | Target Spot (*Corynespora cassiicola*) | ⚠️ Diseased |
| **Tomato** | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Yellow Leaf Curl Virus (TYLCV) | ⚠️ Diseased |
| **Tomato** | `Tomato__Tomato_mosaic_virus` | Mosaic Virus (ToMV) | ⚠️ Diseased |
| **Tomato** | `Tomato_healthy` | Healthy Tomato Foliage | ✅ Healthy |

---

## 🔬 Dataset Inspection & Preparation
The raw dataset was scanned from `D:\` with individual class folders.
- **Total Images Inspected**: 20,639 files across 15 folders.
- **Corrupted Images Detected**: 1 corrupted file in `Tomato__Tomato_YellowLeaf__Curl_Virus` (filtered out).
- **Exact Hash Duplicates Detected**: 14 identical images (8 in Late Blight, 6 in Healthy) filtered via MD5 hashing to prevent data leakage.
- **Total Clean, Unique Images**: **20,625 images**.
- **Standard Image Dimensions**: 256 x 256 pixels, RGB 3-channel.

### Stratified 70 : 15 : 15 Splitting
Using deterministic random seed `42`, every class was split in exact stratified proportion:
- **Training Set (70.01%)**: **14,438 images**
- **Validation Set (15.01%)**: **3,095 images**
- **Held-Out Test Set (14.99%)**: **3,091 images** (Kept strictly isolated for final benchmark evaluation).

---

## 🏗️ Deep Learning Architecture: MobileNetV2
MobileNetV2 was selected as the deep neural network backbone due to its optimal balance of high representational capacity and low computational complexity (~3.5 million parameters), making it ideal for agricultural edge deployment.

### Structural Design:
1. **Input Layer**: `(224, 224, 3)` RGB leaf tensors.
2. **Preprocessing**: MobileNetV2 normalization mapping integer pixels `[0, 255]` to `[-1.0, 1.0]`.
3. **Data Augmentation**: Real-time random horizontal and vertical flips, random rotation (15%), random zoom (15%), and random spatial translation (10%).
4. **Pretrained Base**: MobileNetV2 backbone pretrained on ImageNet (154 layers).
5. **Pooling & Regularization**:
   - `GlobalAveragePooling2D()` (converts 7x7 spatial feature maps into a 1280-d embedding).
   - `BatchNormalization()` (stabilizes gradient dynamics).
   - `Dropout(0.3)` (prevents co-adaptation of neurons and combats overfitting).
6. **Classification Head**: `Dense(15, activation="softmax")` producing a probability distribution across the 15 classes.

### Two-Stage Training Strategy:
- **Phase 1 (Feature Extraction)**: The pretrained convolutional base is frozen (`base_model.trainable = False`). Only the classification head is trained for 8 epochs using Adam optimizer with learning rate $\eta = 10^{-3}$.
- **Phase 2 (Fine-Tuning)**: Layers from index 120 upwards are unfrozen. The model is fine-tuned for 6 epochs using a reduced learning rate $\eta = 10^{-5}$ alongside `ReduceLROnPlateau` and `EarlyStopping` (restoring best weights based on validation loss).

---

## 📊 Actual Experimental Results & Evaluation
The model was evaluated on the **completely untouched 15% held-out test dataset (3,091 images)**.

### Benchmark Summary Metrics:
- **Overall Test Accuracy**: **84.67%**
- **Macro Average Precision**: **82.33%**
- **Macro Average Recall**: **77.45%**
- **Macro Average F1-Score**: **79.16%**
- **Weighted Average Precision**: **84.70%**
- **Weighted Average Recall**: **84.67%**
- **Weighted Average F1-Score**: **84.40%**

### Per-Class Performance Breakdown:
| Class Name | Precision | Recall | F1-Score | Test Samples |
|---|---|---|---|---|
| Pepper Bell Bacterial Spot | 0.8485 | 0.7517 | **0.7972** | 149 |
| Pepper Bell Healthy | 0.8565 | 0.8643 | **0.8604** | 221 |
| Potato Early Blight | 0.8944 | 0.8467 | **0.8699** | 150 |
| Potato Late Blight | 0.8182 | 0.7200 | **0.7660** | 150 |
| Potato Healthy | 0.4000 | 0.1739 | **0.2424** | 23 |
| Tomato Bacterial Spot | 0.8631 | 0.9091 | **0.8855** | 319 |
| Tomato Early Blight | 0.8051 | 0.6333 | **0.7090** | 150 |
| Tomato Late Blight | 0.7529 | 0.8982 | **0.8192** | 285 |
| Tomato Leaf Mold | 0.8819 | 0.7832 | **0.8296** | 143 |
| Tomato Septoria Leaf Spot | 0.7917 | 0.8604 | **0.8246** | 265 |
| Tomato Spider Mites | 0.8555 | 0.8690 | **0.8622** | 252 |
| Tomato Target Spot | 0.7890 | 0.8190 | **0.8037** | 210 |
| Tomato Yellow Leaf Curl Virus | 0.8907 | 0.9148 | **0.9026** | 481 |
| Tomato Mosaic Virus | 0.9444 | 0.6071 | **0.7391** | 56 |
| Tomato Healthy | 0.9582 | 0.9662 | **0.9622** | 237 |

> [!NOTE]
> *Scientific Transparency*: Classes with high sample representations (e.g. Tomato Healthy F1: 96.2%, TYLCV F1: 90.3%, Tomato Bacterial Spot F1: 88.6%) demonstrated exceptional recognition. Potato Healthy has an initial dataset limitation in PlantVillage (only 152 total images available in the original repository), which is honestly reflected in its recall and highlighted as an area for targeted data collection in Part 2.

---

## 🚀 FastAPI Backend Endpoints
Interactive Swagger documentation is automatically hosted at: `http://127.0.0.1:8000/docs`

### 1. `GET /health`
Returns system health, model loading state, and active classification backend.
```json
{
  "status": "healthy",
  "service": "AI-Enabled Crop Disease Detection API",
  "version": "1.0.0",
  "model_status": {
    "loaded": true,
    "backend": "joblib",
    "classes_count": 15,
    "model_version": "1.0.0"
  }
}
```

### 2. `GET /classes`
Returns the complete taxonomy of 15 tracked diseases grouped by crop.
```json
{
  "status": "success",
  "total_classes": 15,
  "crops": ["Pepper (Bell Pepper)", "Potato", "Tomato"],
  "classes_by_crop": { ... },
  "classes": [ ... ]
}
```

### 3. `POST /predict`
Accepts a leaf image file (`multipart/form-data`) and returns the primary diagnosis, top-3 candidates, and agricultural advisory.
**Response Sample**:
```json
{
  "status": "success",
  "filename": "tomato_early_blight.jpg",
  "file_size_kb": 24.8,
  "predicted_crop": "Tomato",
  "predicted_disease": "Early Blight",
  "is_healthy": false,
  "confidence": 95.9,
  "top_predictions": [
    {
      "class_name": "Tomato_Early_blight",
      "crop": "Tomato",
      "disease": "Early Blight",
      "is_healthy": false,
      "confidence": 95.9
    },
    {
      "class_name": "Tomato__Target_Spot",
      "crop": "Tomato",
      "disease": "Target Spot",
      "is_healthy": false,
      "confidence": 3.2
    },
    {
      "class_name": "Tomato_Septoria_leaf_spot",
      "crop": "Tomato",
      "disease": "Septoria Leaf Spot",
      "is_healthy": false,
      "confidence": 0.8
    }
  ],
  "advisory": {
    "crop": "Tomato",
    "condition": "Early Blight",
    "scientific_name": "Alternaria linariae (Alternaria solani)",
    "category": "diseased",
    "symptoms": [
      "Dark brown concentric ring spots ('target board') on older leaves.",
      "Yellowing of surrounding tissue and gradual leaf drop from bottom upwards."
    ],
    "prevention": [
      "Apply organic mulch around plant base to prevent soil-borne spore splashing.",
      "Maintain adequate spacing (60-90 cm) and prune lower suckers to promote air circulation."
    ],
    "management": [
      "Prune off lower infected leaves as soon as first symptoms appear.",
      "Apply protective organic sprays such as copper octanoate or biofungicides."
    ],
    "disclaimer": "DISCLAIMER: This assessment is generated by an Artificial Intelligence deep learning model as an educational and preliminary diagnostic tool..."
  }
}
```

---

## 💻 Installation & Quick Start Guide

### Step 1: Clone or Navigate to Project Directory
```powershell
cd D:\AI-Crop-Disease-Detection-Advisory
```

### Step 2: Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### Step 3: Run One-Click Startup Script
Double-click `run_project.bat` or run:
```powershell
.\run_project.bat
```
The script automatically starts Uvicorn at `http://127.0.0.1:8000` and launches your web browser.

---

## ☁️ Google Colab Training Guide (Free GPU)
To train the heavy MobileNetV2 deep learning network using free cloud GPUs:
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload Notebook** and choose `notebooks/mobilenetv2_training.ipynb`.
3. Set the runtime: **Runtime ➔ Change runtime type ➔ Hardware accelerator ➔ T4 GPU**.
4. Zip your `data/processed` folder or upload the original dataset.
5. Execute the cells top-to-bottom:
   - Cell 1-3: Hardware verification (`nvidia-smi`) and library imports.
   - Cell 4-6: Dataset unzipping, class mapping, and data augmentation.
   - Cell 7-10: MobileNetV2 architecture compilation, Phase 1 (Frozen Base) and Phase 2 (Fine-Tuning).
   - Cell 11: Evaluation on the 15% test set (confusion matrix and classification report).
   - Cell 12: Download `crop_model_artifacts.zip` containing `crop_model.keras`.
6. Unpack the downloaded `.keras` file directly into your local `models/` directory.

---

## 🧪 Automated Testing Suite
Run the 14 automated unit and integration tests:
```powershell
pytest tests/ -v
```
**Test Coverage Includes**:
- ✅ `test_dataset.py`: Class count validation (15 classes), manifest integrity, 70:15:15 split proportions, zero data leakage between train/val/test splits.
- ✅ `test_model.py`: Predictor model loading, confidence scores, probability ordering.
- ✅ `test_api.py`: `GET /health`, `GET /classes`, `POST /predict`, empty upload rejection (HTTP 400), invalid extension rejection (HTTP 400), corrupted image rejection (HTTP 400).
- ✅ `test_all_classes.py`: Actual end-to-end inference tested on held-out images from all 15 classes.

---

## ⚠️ System Limitations & Future Scope
- **Part 1 Scope**: Focused on single-leaf photographic classification under controlled backgrounds.
- **Field Clutter**: Complex multi-leaf backgrounds and shadows in outdoor field conditions can affect feature extraction.
- **Under-represented Classes**: Potato Healthy class has limited sample availability in PlantVillage (152 images), leading to reduced recall.
- **Part 2 Roadmap**:
  - Integration of YOLOv8 object detection for multi-leaf field localization.
  - LLM-powered multi-lingual voice advisory in local regional languages (Hindi, Telugu, Tamil, Marathi).
  - Atmospheric weather API integration to correlate humidity/rainfall with blight vulnerability.

---

## 👨‍💻 Project Academic Metadata
- **Project Title**: AI-Enabled Crop Disease Detection and Advisory System
- **Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering (AI/ML)
- **Component**: Final-Year Major Project (Part 1 Complete Implementation)
- **Dataset Repository**: PlantVillage Pathology Dataset
