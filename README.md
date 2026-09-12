# SIH26139 — Hybrid Quantum-Classical Disease Detection

> **Smart India Hackathon 2026 | Problem Statement SIH26139**

A production-grade hybrid quantum-classical machine learning pipeline for multi-modal disease detection:
- 🧠 **Brain Tumor Classification** (MRI, 4-class)
- ❤️ **Heart Disease Prediction** (Cleveland dataset)
- 🔬 **Breast Cancer Diagnosis** (Wisconsin WBCD)

## Architecture

```
data/           → raw & processed datasets (gitignored)
src/
  preprocessing/  → ingestion, cleaning, feature engineering
  quantum/        → quantum circuits, VQC, QNN layers
  classical/      → baseline models (XGBoost, sklearn, CNN)
  explainability/ → SHAP, GradCAM, Lime
  dashboard/      → Streamlit UI
  api/            → FastAPI inference endpoints
models/           → checkpoints (gitignored)
results/          → metrics, figures
docs/             → reports, design docs
notebooks/        → EDA & experimentation
tests/            → unit & integration tests
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/alihuzaifa-siddiqui302/sih-quantum-disease-detection.git
cd sih-quantum-disease-detection

# 2. Create venv (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ingest data (requires ~/.kaggle/kaggle.json)
python -m src.preprocessing.ingest

# 5. Run tests
pytest tests/ -v
```

## Environment

- Python 3.11
- Qiskit + Qiskit Machine Learning
- PennyLane + Lightning backend
- PyTorch + torchvision
- scikit-learn, XGBoost, SHAP
- Streamlit dashboard + FastAPI
