"""
src/explainability/shap_tabular.py
===================================
SHAP explainability pipeline for tabular clinical models (Heart Disease & WDBC).
Computes Shapley attribution values in latent PCA space and maps them back
to native clinical biomarkers by inverting the linear PCA projection matrix:
    Φ_native = Φ_latent · W_pca

Produces:
    results/figures/shap_heart_biomarkers.png
    results/figures/shap_wdbc_biomarkers.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
PROCESSED_DIR  = ROOT / "data" / "processed"
RAW_DIR        = ROOT / "data" / "raw"
FIGURES_DIR    = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HEART_FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results (0-2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=yes, 0=no)",
    "oldpeak": "ST Depression Induced by Exercise",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Number of Major Vessels (0-3) by Fluoroscopy",
    "thal": "Thallium Stress Scintigraphy",
}


def explain_tabular_dataset(dataset: str, out_filename: str):
    print(f"\n[SHAP] Processing dataset: {dataset.upper()}...")

    # 1. Load pipeline artifacts and processed data
    pipe_path = CHECKPOINT_DIR / f"{dataset}_tabular_pipeline.pkl"
    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline artifact missing: {pipe_path}")
    pipeline_obj = joblib.load(pipe_path)
    pca = pipeline_obj["pca"]
    sel_indices = pipeline_obj["selected_feature_indices"]
    k_pca = pipeline_obj.get("k_pca", pca.n_components_)
    W_pca = pca.components_[:k_pca, :]  # [k_pca, n_selected]

    # Load raw feature names
    csv_file = RAW_DIR / ("heart_disease.csv" if dataset == "heart" else "wbcd.csv")
    df_raw = pd.read_csv(csv_file)
    raw_feature_names = [c for c in df_raw.columns if c != "target"]
    selected_feature_names = [raw_feature_names[i] for i in sel_indices]

    # Load test split
    npz_data = np.load(PROCESSED_DIR / f"{dataset}_processed.npz")
    X_test = npz_data["X_test"]
    X_train = npz_data["X_train"]

    # 2. Load trained Tree model (Random Forest or XGBoost)
    model_path = CHECKPOINT_DIR / f"classical_rf_{dataset}.pkl"
    if not model_path.exists():
        model_path = CHECKPOINT_DIR / f"classical_xgb_{dataset}.pkl"
    clf = joblib.load(model_path)

    # 3. Compute SHAP values in latent PCA space
    explainer = shap.TreeExplainer(clf)
    shap_vals_raw = explainer.shap_values(X_test)

    # Handle binary classification format across shap versions
    if isinstance(shap_vals_raw, list):
        # Class 1 SHAP values (disease present)
        phi_latent = shap_vals_raw[1]
    elif isinstance(shap_vals_raw, np.ndarray):
        if shap_vals_raw.ndim == 3:
            phi_latent = shap_vals_raw[:, :, 1]
        else:
            phi_latent = shap_vals_raw
    else:
        phi_latent = np.array(shap_vals_raw)

    # 4. Invert linear projection: phi_native = phi_latent @ W_pca
    # phi_latent: [N, k], W_pca: [k, n_sel] -> phi_native: [N, n_sel]
    phi_native = np.dot(phi_latent, W_pca)

    # Calculate mean absolute impact per biomarker
    mean_abs_impact = np.mean(np.abs(phi_native), axis=0)
    sorted_idx = np.argsort(mean_abs_impact)[::-1]

    sorted_names = [selected_feature_names[i] for i in sorted_idx]
    sorted_impacts = mean_abs_impact[sorted_idx]

    # Generate friendly labels
    labels = []
    for name in sorted_names:
        if dataset == "heart" and name in HEART_FEATURE_DESCRIPTIONS:
            labels.append(f"{name} ({HEART_FEATURE_DESCRIPTIONS[name]})")
        else:
            labels.append(name.replace("_", " ").title())

    # 5. Create publication-quality dual visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1]})
    plt.subplots_adjust(wspace=0.35)

    # Subplot 1: Bar chart of Global Biomarker Importance
    y_pos = np.arange(len(sorted_impacts))[::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sorted_impacts)))

    ax1.barh(y_pos, sorted_impacts, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Mean |SHAP Value| (Impact on Disease Prediction)", fontsize=11, fontweight="bold")
    title_suffix = "Heart Disease (UCI 45)" if dataset == "heart" else "Breast Cancer (WDBC)"
    ax1.set_title(f"Global Biomarker Attribution: {title_suffix}\n(PCA Inverse-Mapped from Latent Space)", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate values
    for i, v in enumerate(sorted_impacts):
        ax1.text(v + (max(sorted_impacts) * 0.015), y_pos[i], f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

    # Subplot 2: Native Feature Impact Distribution (Synthetic Beeswarm representation)
    phi_sorted = phi_native[:, sorted_idx]
    for i in range(len(sorted_names)):
        vals = phi_sorted[:, i]
        y_scatter = np.full_like(vals, y_pos[i]) + np.random.normal(0, 0.08, size=len(vals))
        ax2.scatter(vals, y_scatter, alpha=0.6, s=25, c=vals, cmap="coolwarm", edgecolors="none")

    ax2.axvline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s.title() for s in sorted_names], fontsize=10)
    ax2.set_xlabel("SHAP Value (Directional Impact on Disease Risk)", fontsize=11, fontweight="bold")
    ax2.set_title("Biomarker Impact Distribution across Test Patients\n(Red = Positive Risk, Blue = Protective)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle(f"Clinical Explainability Pipeline: {title_suffix}\nInverting Quantum-Classical Latent Projections to Native Diagnostics", fontsize=14, fontweight="bold", y=0.98)

    out_path = FIGURES_DIR / out_filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Successfully generated {out_path}")


def run_shap_pipeline():
    explain_tabular_dataset("heart", "shap_heart_biomarkers.png")
    explain_tabular_dataset("wbcd", "shap_wdbc_biomarkers.png")


if __name__ == "__main__":
    run_shap_pipeline()
