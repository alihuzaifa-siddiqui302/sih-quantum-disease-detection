# SIH26139 — Hybrid Quantum-Classical Disease Detection

> **Smart India Hackathon 2026 | Problem Statement SIH26139**  
> A multi-modal hybrid quantum-classical machine learning system for precision medical diagnostics, clinical explainability, and risk stratification.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Datasets & Automated Ingestion](#-datasets--automated-ingestion)
- [Repository Structure](#-repository-structure)
- [Quick Start & Environment Setup](#-quick-start--environment-setup)
- [End-to-End Reproduction Guide](#-end-to-end-reproduction-guide)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Tabular Preprocessing](#2-tabular-preprocessing)
  - [3. Classical Baselines](#3-classical-baselines)
  - [4. Quantum Model Training](#4-quantum-model-training)
  - [5. Master Benchmark Synthesis & Verification](#5-master-benchmark-synthesis--verification)
  - [6. Clinical Explainability Suite](#6-clinical-explainability-suite)
  - [7. Interactive Clinical Platform & API](#7-interactive-clinical-platform--api)
- [Published Benchmark Ground Truth](#-published-benchmark-ground-truth)
- [Automated Test Suite](#-automated-test-suite)

---

## 🔬 Overview

This repository implements a modular clinical diagnostic pipeline combining Parameterized Quantum Circuits (PQCs), Quantum Support Vector Classifiers (QSVC), and classical baselines across three clinical modalities:

1. 🧠 **Brain Tumor Classification (Imaging, 4-class):**
   * **Model:** Hybrid Quantum Transfer Learning (QTL) featuring a frozen/fine-tuned ResNet-18 backbone linked to a 6-qubit Parameterized Quantum Circuit (`AngleEmbedding` + `StronglyEntanglingLayers` depth 3, PennyLane Lightning statevector backend).
   * **Classes:** Glioma, Meningioma, Pituitary, No Tumor.
2. ❤️ **Coronary Heart Disease (Tabular):**
   * **Model:** Quantum Support Vector Classifier (QSVC) using a 6-qubit `ZZFeatureMap` ($reps=2$) with `FidelityQuantumKernel` on Qiskit Aer, compared against Logistic Regression, SVM-RBF, Random Forest, and XGBoost.
3. 🔬 **Breast Cancer Diagnosis (Cytopathology / WBCD):**
   * **Model:** 5-qubit QSVC kernel classifier evaluated under both noiseless and IBM 7-qubit calibrated thermal/depolarizing noise profiles, featuring high-sensitivity triage thresholding ($\tau=0.10$).

---

## 📦 Datasets & Automated Ingestion

All datasets are ingested programmatically via `src/preprocessing/ingest.py`.

| Dataset | Modality | Exact Source Identifier | Samples / Split | Ingestion Target |
| :--- | :--- | :--- | :--- | :--- |
| **Brain Tumor MRI** | MRI (224×224) | Kaggle: `masoudnickparvar/brain-tumor-mri-dataset` | 7,023+ images (4 classes) | `data/raw/brain_mri/` |
| **Heart Disease** | Clinical Tabular | UCI ML Repository: `id=45` (Cleveland) | 303 patients (13 features) | `data/raw/heart_disease.csv` |
| **Breast Cancer (WBCD)** | Cytopathology | Scikit-learn: `sklearn.datasets.load_breast_cancer` | 569 patients (30 features) | `data/raw/wbcd.csv` |

### Kaggle API Setup (Required for Brain MRI)
To download the Brain MRI dataset automatically:
1. Log in to [Kaggle](https://www.kaggle.com) $\to$ **Account Settings** $\to$ **Create New API Token** (`kaggle.json`).
2. Place `kaggle.json` in:
   * **Linux/macOS:** `~/.kaggle/kaggle.json`
   * **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`
3. Secure permissions (Linux/macOS): `chmod 600 ~/.kaggle/kaggle.json`

---

## 📂 Repository Structure

```text
sih-quantum/
├── data/                       # Datasets (gitignored)
│   ├── raw/                    # Downloaded raw datasets (brain_mri, heart_disease.csv, wbcd.csv)
│   └── processed/              # Stratified .npz splits, scaler/PCA pipelines, feature caches
├── docs/                       # Verification and audit documentation
├── models/
│   ├── checkpoints/            # Saved weights (.pth, .pkl, .npz)
│   └── qtl_architecture.py     # PennyLane 6-qubit QTL TorchLayer & hybrid PyTorch model
├── platform/
│   ├── backend/
│   │   └── main.py             # Production FastAPI clinical inference server
│   └── frontend/
│       ├── app.py              # Streamlit clinical diagnostic dashboard
│       ├── index.html          # Standalone responsive glassmorphic medical portal
│       └── assets/             # Synchronized explainability figures and plots
├── results/
│   ├── figures/                # Visualizations (Grad-CAM, SHAP, training curves, sensitivity)
│   └── metrics/                # Machine-readable benchmark JSONs, reports, and raw .npy test predictions
├── src/
│   ├── preprocessing/          # Ingestion, stratified splitting, PCA projection, augmentation
│   ├── classical/              # Tabular baselines (LR, SVM, RF, XGB) & imaging ResNet-18
│   ├── quantum/                # QSVC engine (Qiskit) & QTL training loops (PennyLane)
│   └── explainability/         # Grad-CAM heatmaps, inverted-PCA SHAP, quantum sensitivity
└── tests/                      # Pytest verification suites (scaffold, quantum, benchmark, API)
```

> **Note on Navigation:** Production API endpoints and interactive UI applications reside in `platform/backend/` and `platform/frontend/`. Subdirectories `src/api/` and `src/dashboard/` contain pointers to the `platform/` modules.

---

## ⚡ Quick Start & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/alihuzaifa-siddiqui302/sih-quantum-disease-detection.git
cd sih-quantum-disease-detection
```

### 2. Create and Activate Virtual Environment (Python 3.11)

**Windows (PowerShell):**
```powershell
$env:PYTHONUTF8="1"
py -3.11 -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
export PYTHONUTF8=1
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Dependencies are strictly pinned in `requirements.txt` (encoded in standard UTF-8):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Project Scaffold
Ensure directory structure, packages, and environment gates are intact:
```bash
pytest tests/test_scaffold.py -v
```

---

## 🚀 End-to-End Reproduction Guide

Follow these sequential steps to ingest data, train baselines and quantum models, regenerate benchmarks, and validate test metrics.

### 1. Data Ingestion
Download and organize all three datasets:
```bash
# Ingest all three datasets
python -m src.preprocessing.ingest

# Or ingest individually:
python -m src.preprocessing.ingest --source mri
python -m src.preprocessing.ingest --source heart
python -m src.preprocessing.ingest --source wbcd
```

### 2. Tabular Preprocessing
Clean tabular datasets, perform mutual-information feature ranking, apply PCA bottleneck projection ($k=6$ for Heart, $k=5$ for WBCD), and save stratified splits ($70\%$ train, $15\%$ val, $15\%$ test):
```bash
python -m src.preprocessing.tabular_pipeline
```
Outputs saved to `data/processed/{heart,wbcd}_processed.npz` and `models/checkpoints/{heart,wbcd}_tabular_pipeline.pkl`.

### 3. Classical Baselines
Train all classical tabular baselines (Logistic Regression, SVM-RBF, Random Forest, XGBoost) with cross-validated hyperparameter grids:
```bash
# Train tabular baselines for both Heart and WBCD
python -m src.classical.baseline_tabular

# Train pure classical ResNet-18 baseline on Brain MRI (mirrors QTL two-stage training)
python -m src.classical.baseline_imaging
```

### 4. Quantum Model Training

#### A. Quantum Support Vector Classifier (QSVC)
Train QSVC models with `ZZFeatureMap` kernel matrices and validation $F_2$-guided $C$-sweeps:
```bash
# Run noiseless simulations for both Heart and WBCD (~5-15 min on CPU)
python -m src.quantum.train_qsvc --modes noiseless

# (Optional) Run noisy simulation on Heart Disease with IBM 7-qubit calibrated noise model
python -m src.quantum.train_qsvc --dataset heart --modes noisy
```
> **Note on Runtime:** Noisy simulation on WBCD requires substantial CPU compute ($>40$ minutes). The benchmark suite automatically records a compliant timeout stub (`"status": "DNF"`) if skipped.

#### B. Quantum Transfer Learning (QTL)
Train the 6-qubit hybrid ResNet-18 + PQC network on Brain MRI using CPU feature caching:
```bash
# Full Stage 1 (frozen backbone warmup) + Stage 2 (joint fine-tuning with differential LRs)
python -m src.quantum.train_qtl --seed 42

# Or run a rapid evaluation test:
python -m src.quantum.train_qtl --epochs1 5 --epochs2 5 --seed 42
```
* Best Stage 1 weights: `models/checkpoints/qtl_stage1_best.pth`
* Best Stage 2 weights: `models/checkpoints/qtl_best.pth`
* Training curves: `results/figures/training_curves_qtl.png`

### 5. Master Benchmark Synthesis & Verification

Regenerate consolidated metrics, compute McNemar's statistical significance tests ($p$-values), and evaluate quantum circuit efficiency:
```bash
# Synthesize results/metrics/benchmark.json and benchmark_report.md
python -m src.classical.benchmark

# Run the 34-gate automated benchmark verification test suite
pytest tests/test_benchmark.py -v
```

### 6. Clinical Explainability Suite

Generate visual explanations for clinical interpretability:
```bash
# 1. Grad-CAM saliency overlays comparing classical ResNet-18 vs QTL
python -m src.explainability.gradcam

# 2. Inverted-PCA SHAP biomarker importance for Heart Disease and Breast Cancer
python -m src.explainability.shap_tabular

# 3. 6-qubit bottleneck Jacobian sensitivity matrix |∂⟨Z_j⟩/∂z_k|
python -m src.explainability.quantum_sensitivity
```
Figures are saved in `results/figures/` and mirrored to `platform/frontend/assets/`.

### 7. Interactive Clinical Platform & API

#### Launch the FastAPI Clinical Backend
```bash
uvicorn platform.backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger docs available at: `http://127.0.0.1:8000/docs`

#### Run Automated Backend Integration Tests
In a separate terminal, test all endpoints (`/model-info`, `/results-summary`, `/predict`, `/predict-image`):
```bash
python -m tests.test_integration_backend
```

#### Launch the Streamlit Clinical Frontend
```bash
streamlit run platform.frontend.app.py
```
Dashboard available at: `http://127.0.0.1:8501`

#### Standalone Static Web Portal
Open `platform/frontend/index.html` in any modern web browser for a zero-dependency medical portal connected to the live backend.

---

## 📊 Published Benchmark Ground Truth

Reruns can be directly validated against the verified ground truth saved in [`results/metrics/benchmark_report.md`](file:///c:/Users/ADMIN/Desktop/sih-quantum/results/metrics/benchmark_report.md):

### Tabular: Cleveland Heart Disease ($N=303$)
| Model | Accuracy | Precision | Recall | Specificity | Macro $F_1$ | Macro $F_2$ | ROC-AUC | MCC | McNemar vs. Baseline ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QSVC (Noiseless)** | 60.87% | 0.6520 | 0.5790 | 0.5790 | 0.5379 | 0.5510 | N/A | 0.2192 | $p=0.0291$ (vs. LR) |
| **QSVC (Noisy, IBM 7Q)**| 65.22% | 0.6997 | 0.6267 | 0.6267 | 0.6043 | 0.6079 | N/A | 0.3181 | — |
| **Logistic Regression** | **84.78%**| 0.8504 | 0.8524 | 0.8524 | **0.8478** | **0.8497** | **0.9524**| **0.7028**| Baseline |
| **SVM (RBF Kernel)** | **84.78%**| 0.8466 | 0.8486 | 0.8486 | 0.8472 | 0.8479 | 0.8933 | 0.6952 | — |
| **Random Forest** | **84.78%**| 0.8466 | 0.8486 | 0.8486 | 0.8472 | 0.8479 | 0.9295 | 0.6952 | — |
| **XGBoost** | **84.78%**| 0.8466 | 0.8486 | 0.8486 | 0.8472 | 0.8479 | 0.9095 | 0.6952 | — |

### Tabular: Wisconsin Breast Cancer ($N=569$)
| Model | Accuracy | Precision | Recall | Specificity | Macro $F_1$ | Macro $F_2$ | ROC-AUC | MCC | McNemar vs. Baseline ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QSVC (Noiseless)** | 62.79% | 0.3140 | 0.5000 | 0.5000 | 0.3857 | 0.4470 | N/A | 0.0000 | $p=1.4\times 10^{-5}$ (vs. SVM) |
| **SVM (RBF Kernel)** | **90.70%**| **0.9126**| **0.8877**| **0.8877**| **0.8976** | **0.8911** | **0.9780**| **0.7999**| Baseline |
| **Logistic Regression** | 89.53% | 0.9036 | 0.8721 | 0.8721 | 0.8839 | 0.8760 | 0.9682 | 0.7751 | — |
| **XGBoost** | 89.53% | 0.9036 | 0.8721 | 0.8721 | 0.8839 | 0.8760 | 0.9635 | 0.7751 | — |
| **Random Forest** | 88.37% | 0.8861 | 0.8628 | 0.8628 | 0.8720 | 0.8660 | 0.9728 | 0.7486 | — |

### Imaging: Brain Tumor MRI (4-class, Held-out Test $N=1,600$)
| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | Specificity | Macro $F_1$ | Macro $F_2$ | MCC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QTL (Hybrid Quantum)** | **91.00%** | **0.9143** | **0.9100** | **0.9700** | **0.9079** | **0.9083** | **0.8825** | ~80 ms / batch |
| **Classical ResNet-18** | 94.19% | 0.9443 | 0.9419 | 0.9806 | 0.9407 | 0.9409 | 0.9239 | ~32 ms / batch |

*McNemar's test (QTL vs. Classical ResNet-18): $\chi^2 = 38.46, p < 10^{-8}$ (statistically significant differential).*

### Quantum Circuit Architectures
| Engine | Quantum Target | Qubits | Circuit Depth | Total Gates | Trainable Quantum Params |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **QSVC Heart** | Tabular Feature Map | 6 | 49 | 114 | 0 (Fixed dual optimization) |
| **QSVC WBCD** | Tabular Feature Map | 5 | 40 | 80 | 0 (Fixed dual optimization) |
| **QTL Hybrid** | MRI Latent Bottleneck | 6 | 3 layers | Strongly Entangling | 54 ($6 \times 3 \times 3$) |

---

## 🧪 Automated Test Suite

The repository features 68 verified automated test assertions:

```bash
# 1. Project scaffold & integrity
pytest tests/test_scaffold.py -v

# 2. Quantum engines & circuit generation
pytest tests/test_quantum_engines.py -v

# 3. Master benchmark gates (34 assertions)
pytest tests/test_benchmark.py -v

# 4. Run all unit & integration tests
pytest tests/ -v
```

---
*SIH26139 — Hybrid Quantum-Classical Disease Detection Pipeline*
