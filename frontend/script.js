/**
 * AgriGuard AI - Frontend Controller
 * Connects UI with FastAPI backend (/health, /classes, /predict)
 */

document.addEventListener("DOMContentLoaded", () => {
    // API Endpoints
    const API_BASE = window.location.origin;
    const ENDPOINTS = {
        health: `${API_BASE}/health`,
        classes: `${API_BASE}/classes`,
        predict: `${API_BASE}/predict`
    };

    // DOM Elements
    const backendStatus = document.getElementById("backendStatus");
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropPrompt = document.getElementById("dropPrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const fileMeta = document.getElementById("fileMeta");
    const clearBtn = document.getElementById("clearBtn");
    const predictBtn = document.getElementById("predictBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const errorAlert = document.getElementById("errorAlert");
    const errorMsg = document.getElementById("errorMsg");

    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const resDiseaseName = document.getElementById("resDiseaseName");
    const resCropName = document.getElementById("resCropName");
    const resHealthStatus = document.getElementById("resHealthStatus");
    const resConfidenceVal = document.getElementById("resConfidenceVal");
    const topPredictionsList = document.getElementById("topPredictionsList");
    const advSymptoms = document.getElementById("advSymptoms");
    const advPrevention = document.getElementById("advPrevention");
    const advManagement = document.getElementById("advManagement");
    const advDisclaimer = document.getElementById("advDisclaimer");
    const classesContainer = document.getElementById("classesContainer");

    let currentFile = null;

    // 1. Check Backend & Model Health
    async function checkHealth() {
        try {
            const res = await fetch(ENDPOINTS.health);
            if (!res.ok) throw new Error("Server responded with error");
            const data = await res.json();
            
            if (data.model_status && data.model_status.loaded) {
                backendStatus.innerHTML = `
                    <span class="status-dot online"></span>
                    <span class="status-text">API Online • Model Ready (${data.model_status.backend})</span>
                `;
            } else {
                backendStatus.innerHTML = `
                    <span class="status-dot pulsing"></span>
                    <span class="status-text">API Online • Model Loading</span>
                `;
            }
        } catch (err) {
            backendStatus.innerHTML = `
                <span class="status-dot offline"></span>
                <span class="status-text">Backend Offline</span>
            `;
        }
    }

    // 2. Load and Display Supported Classes
    async function loadSupportedClasses() {
        try {
            const res = await fetch(ENDPOINTS.classes);
            if (!res.ok) throw new Error("Could not load classes");
            const data = await res.json();

            if (data.status === "success" && data.classes_by_crop) {
                let html = '<div class="classes-grid">';
                for (const [crop, classes] of Object.entries(data.classes_by_crop)) {
                    html += `
                        <div class="crop-card">
                            <div class="crop-card-header">
                                <span style="font-size: 1.5rem;">🌿</span>
                                <div>
                                    <h3>${crop}</h3>
                                    <small style="color: var(--text-muted);">${classes.length} conditions tracked</small>
                                </div>
                            </div>
                            <div class="class-tag-list">
                    `;
                    classes.forEach(c => {
                        const statusClass = c.is_healthy ? "healthy" : "diseased";
                        const icon = c.is_healthy ? "✓" : "⚠️";
                        html += `
                            <div class="class-pill ${statusClass}">
                                <span>${c.condition}</span>
                                <span style="font-size: 0.8rem;">${icon}</span>
                            </div>
                        `;
                    });
                    html += `
                            </div>
                        </div>
                    `;
                }
                html += '</div>';
                classesContainer.innerHTML = html;
            }
        } catch (err) {
            classesContainer.innerHTML = `<div class="classes-loading" style="color: var(--accent-rose);">Could not load supported classes catalog from server.</div>`;
        }
    }

    // 3. File Selection & Drag-and-Drop Handlers
    browseBtn.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", (e) => {
        if (e.target !== clearBtn && !currentFile) {
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        hideError();
        const validTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!validTypes.includes(file.type)) {
            showError("Unsupported file type. Please upload a JPEG, PNG, or WEBP image.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError("File size exceeds 10 MB limit.");
            return;
        }

        currentFile = file;

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            const sizeKB = (file.size / 1024).toFixed(1);
            fileMeta.textContent = `${file.name} • ${sizeKB} KB`;

            dropPrompt.style.display = "none";
            previewContainer.style.display = "block";
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropPrompt.style.display = "flex";
        previewContainer.style.display = "none";
        predictBtn.disabled = true;
        hideError();
    }

    function showError(message) {
        errorMsg.textContent = message;
        errorAlert.style.display = "flex";
    }

    function hideError() {
        errorAlert.style.display = "none";
        errorMsg.textContent = "";
    }

    // 4. Predict Handler
    predictBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        hideError();
        predictBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        predictBtn.querySelector(".btn-text").textContent = "Analyzing Leaf Pathology...";

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const res = await fetch(ENDPOINTS.predict, {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || "Prediction request failed.");
            }

            displayResults(data);
        } catch (err) {
            showError(err.message || "Failed to communicate with prediction service.");
        } finally {
            predictBtn.disabled = false;
            btnSpinner.style.display = "none";
            predictBtn.querySelector(".btn-text").textContent = "Diagnose Leaf Disease";
        }
    });

    // 5. Render Results View
    function displayResults(data) {
        resultPlaceholder.style.display = "none";
        resultContent.style.display = "block";

        resDiseaseName.textContent = data.predicted_disease;
        resCropName.textContent = data.predicted_crop;

        if (data.is_healthy) {
            resHealthStatus.textContent = "Healthy Condition";
            resHealthStatus.className = "badge badge-healthy";
        } else {
            resHealthStatus.textContent = "Pathological Disease";
            resHealthStatus.className = "badge badge-diseased";
        }

        resConfidenceVal.textContent = `${data.confidence.toFixed(1)}%`;

        // Render Top 3 Bars
        topPredictionsList.innerHTML = "";
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach(p => {
                const item = document.createElement("div");
                item.className = "prediction-item";
                item.innerHTML = `
                    <div class="prediction-meta">
                        <span>${p.crop} — ${p.disease}</span>
                        <span>${p.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="prediction-bar-bg">
                        <div class="prediction-bar-fill" style="width: ${p.confidence}%;"></div>
                    </div>
                `;
                topPredictionsList.appendChild(item);
            });
        }

        // Render Advisory Details
        const adv = data.advisory || {};

        advSymptoms.innerHTML = "";
        (adv.symptoms || ["No symptom data available."]).forEach(s => {
            const li = document.createElement("li");
            li.textContent = s;
            advSymptoms.appendChild(li);
        });

        advPrevention.innerHTML = "";
        (adv.prevention || ["Follow standard agricultural hygiene."]).forEach(p => {
            const li = document.createElement("li");
            li.textContent = p;
            advPrevention.appendChild(li);
        });

        advManagement.innerHTML = "";
        (adv.management || ["Consult agronomic extension officer."]).forEach(m => {
            const li = document.createElement("li");
            li.textContent = m;
            advManagement.appendChild(li);
        });

        advDisclaimer.textContent = adv.disclaimer || "";

        // Smooth scroll on mobile
        if (window.innerWidth < 900) {
            document.getElementById("resultCard").scrollIntoView({ behavior: "smooth" });
        }
    }

    // Initialize Page
    checkHealth();
    loadSupportedClasses();
});
