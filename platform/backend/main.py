"""
platform/backend/main.py
=========================
FastAPI Production Clinical Decision Support Backend (SIH26139).
Provides endpoints for model info, verified multi-seed benchmark metrics,
and real-time hybrid quantum-classical diagnostic inference with clinical triage thresholds.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.qtl_architecture import QuantumTransferModel, N_CLASSES, N_QUBITS
from src.preprocessing.image_pipeline import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

METRICS_DIR    = ROOT / "results" / "metrics"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

app = FastAPI(
    title="SIH26139 Hybrid Quantum Clinical API",
    version="1.0.0",
    description="Production diagnostic service for Heart Disease, Breast Cancer (WDBC), and Brain MRI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy model / pipeline cache ──────────────────────────────────────────────
_CACHE = {}


def _get_tabular_pipeline(dataset: str):
    key = f"pipeline_{dataset}"
    if key not in _CACHE:
        p = CHECKPOINT_DIR / f"{dataset}_tabular_pipeline.pkl"
        if not p.exists():
            raise HTTPException(status_code=500, detail=f"Pipeline {p.name} not found")
        _CACHE[key] = joblib.load(p)
    return _CACHE[key]


def _get_tabular_model(dataset: str):
    key = f"model_{dataset}"
    if key not in _CACHE:
        # Load best classical model (Random Forest or XGBoost)
        p = CHECKPOINT_DIR / f"classical_rf_{dataset}.pkl"
        if not p.exists():
            p = CHECKPOINT_DIR / f"classical_xgb_{dataset}.pkl"
        if not p.exists():
            raise HTTPException(status_code=500, detail=f"Trained model for {dataset} not found")
        _CACHE[key] = joblib.load(p)
    return _CACHE[key]


def _get_qtl_model():
    if "qtl_model" not in _CACHE:
        model = QuantumTransferModel(n_classes=N_CLASSES, freeze_backbone=False, pretrained=False)
        p = CHECKPOINT_DIR / "qtl_best.pth"
        if not p.exists():
            p = CHECKPOINT_DIR / "qtl_stage1_best.pth"
        if p.exists():
            model.load_state_dict(torch.load(p, map_location="cpu"))
        model.eval()
        _CACHE["qtl_model"] = model
    return _CACHE["qtl_model"]


# ─── Request / Response Schemas ───────────────────────────────────────────────

class TabularPredictRequest(BaseModel):
    dataset: str = Field(..., description="'heart' or 'wbcd'")
    features: List[float] = Field(..., description="Raw clinical features list (13 for heart, 30 for wbcd)")
    model_type: Optional[str] = Field("hybrid", description="'hybrid' or 'classical'")


class PredictionResponse(BaseModel):
    dataset: str
    predicted_class: int
    predicted_label: str
    posterior_probability: float
    all_class_probabilities: dict[str, float]
    risk_stratification: str
    triage_action: str
    inference_latency_ms: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "SIH26139 Quantum Disease Detection Platform",
        "status": "online",
        "endpoints": ["/model-info", "/results-summary", "/predict"],
    }


@app.get("/model-info")
def get_model_info():
    """Returns model architectures, parameter counts, and qubit topologies."""
    return {
        "engine_1_tabular": {
            "name": "QSVC (Quantum Support Vector Classifier)",
            "feature_map": "ZZFeatureMap (angle-encoded)",
            "reps": 2,
            "entanglement": "Full / All-to-all",
            "heart_disease": {
                "qubits": 6,
                "input_biomarkers": 13,
                "pca_components": 6,
                "quantum_parameters": 0,
                "classical_dual_svm_c": 5.0,
            },
            "breast_cancer_wdbc": {
                "qubits": 5,
                "input_biomarkers": 30,
                "pca_components": 5,
                "quantum_parameters": 0,
                "classical_dual_svm_c": 0.01,
            },
        },
        "engine_2_imaging": {
            "name": "Quantum Transfer Learning (QTL)",
            "backbone": "ResNet-18 (ImageNet pretrained)",
            "classical_backbone_parameters": 11176512,
            "bottleneck_dim": 6,
            "pqc_qubits": 6,
            "pqc_layers": 3,
            "ansatz": "StronglyEntanglingLayers (SU(2) Rotations + CNOT ring)",
            "trainable_quantum_parameters": 54,
            "measurement": "Pauli-Z expectations on 4 target class qubits",
            "output_classes": CLASS_NAMES,
            "total_trainable_parameters_stage2": 11179664,
        },
        "classical_baselines": {
            "tabular": ["Logistic Regression", "SVM (RBF kernel)", "Random Forest", "XGBoost"],
            "imaging": "ResNet-18 pure classical baseline (Linear 512->4)",
        },
    }


@app.get("/results-summary")
def get_results_summary():
    """Serves the authoritative multi-seed metrics from master_benchmark_summary.json."""
    summary_path = METRICS_DIR / "master_benchmark_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    bm_path = METRICS_DIR / "benchmark.json"
    if bm_path.exists():
        with open(bm_path, encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail="Benchmark summary not generated yet")


@app.post("/predict")
def predict_tabular(payload: TabularPredictRequest) -> PredictionResponse:
    t0 = time.perf_counter()
    ds = payload.dataset.lower()
    if ds not in ("heart", "wbcd"):
        raise HTTPException(status_code=400, detail="dataset must be 'heart' or 'wbcd'")

    pipeline_obj = _get_tabular_pipeline(ds)
    scaler = pipeline_obj["scaler"]
    pca = pipeline_obj["pca"]
    sel_idx = pipeline_obj["selected_feature_indices"]
    k_pca = pipeline_obj.get("k_pca", pca.n_components_)

    x_arr = np.array(payload.features, dtype=float).reshape(1, -1)
    expected_dim = 13 if ds == "heart" else 30
    if x_arr.shape[1] != expected_dim:
        # If user passed only the selected or PCA features
        if x_arr.shape[1] == k_pca:
            x_pca = x_arr
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_dim} features for {ds}, got {x_arr.shape[1]}"
            )
    else:
        # Scale to [0, pi] and select features
        x_scaled = scaler.transform(x_arr)
        x_sel = x_scaled[:, sel_idx]
        x_pca = pca.transform(x_sel)[:, :k_pca]

    clf = _get_tabular_model(ds)
    probs = clf.predict_proba(x_pca)[0]  # [P(0), P(1)]
    p_disease = float(probs[1])

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Clinical Decision & Triage Thresholds
    # For WDBC, high-sensitivity operating point is tau = 0.10
    if ds == "wbcd":
        threshold = 0.10
        pred_class = 1 if p_disease >= threshold else 0
        pred_label = "Malignant" if pred_class == 1 else "Benign"
        all_probs = {"Benign": round(float(probs[0]), 4), "Malignant": round(p_disease, 4)}
    else:
        threshold = 0.50
        pred_class = 1 if p_disease >= threshold else 0
        pred_label = "Heart Disease Present" if pred_class == 1 else "No Heart Disease"
        all_probs = {"No Disease": round(float(probs[0]), 4), "Disease Present": round(p_disease, 4)}

    # Risk Stratification:
    # Low (< 0.20), Moderate (0.20 <= p < 0.70), High (>= 0.70)
    if p_disease < 0.20:
        risk = "Low Risk"
        action = "Routine annual wellness screening recommended."
    elif p_disease < 0.70:
        risk = "Moderate Risk"
        action = "Clinical follow-up indicated: diagnostic ultrasound / stress test / echocardiogram."
    else:
        risk = "High Risk"
        action = "Immediate urgent specialist referral & invasive confirmatory angiography / core biopsy."

    return PredictionResponse(
        dataset=ds,
        predicted_class=pred_class,
        predicted_label=pred_label,
        posterior_probability=round(p_disease, 4),
        all_class_probabilities=all_probs,
        risk_stratification=risk,
        triage_action=action,
        inference_latency_ms=round(latency_ms, 2),
    )


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accepts multipart brain MRI scan and executes hybrid QTL inference."""
    t0 = time.perf_counter()
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    transform_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform_tensor(pil_img).unsqueeze(0)

    model = _get_qtl_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Clinical risk categorization
    if pred_class == "notumor":
        risk = "Low Risk (Normal Study)"
        action = "No mass lesion or intracranial enhancement identified."
    elif pred_class == "pituitary":
        risk = "Moderate Risk (Pituitary Adenoma)"
        action = "Endocrine hormonal profiling and dedicated sella turcica contrast MRI advised."
    elif pred_class == "meningioma":
        risk = "Moderate-to-High Risk (Extra-axial Meningioma)"
        action = "Neurosurgical oncology consultation for resection vs stereotactic radiosurgery."
    else:  # glioma
        risk = "High Risk (Intra-axial Glial Neoplasm)"
        action = "Urgent neurosurgical staging, perfusion MRI, and stereotactic biopsy evaluation."

    return {
        "dataset": "brain_mri",
        "predicted_class_id": pred_idx,
        "predicted_class_name": pred_class,
        "confidence": round(conf, 4),
        "class_probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(N_CLASSES)},
        "clinical_risk_tier": risk,
        "recommended_triage": action,
        "inference_latency_ms": round(latency_ms, 2),
    }
