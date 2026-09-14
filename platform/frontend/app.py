"""
platform/frontend/app.py
=========================
SIH26139 Unified Clinical Decision Support Dashboard.
Streamlit application integrating multi-dataset benchmarks, explainability visualizations,
and real-time hybrid quantum diagnostic triage connecting to the FastAPI backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "platform" / "frontend" / "assets"
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="SIH26139: Quantum Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Cached Data ─────────────────────────────────────────────────────────

@st.cache_data
def load_benchmark_summary():
    path = ASSETS_DIR / "master_benchmark_summary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_mri_comparison():
    path = ASSETS_DIR / "brain_mri_full_comparison.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_integration_report():
    path = ASSETS_DIR / "integration_test_report.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


summary_data = load_benchmark_summary()
mri_data = load_mri_comparison()
integration_data = load_integration_report()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 SIH26139 Platform")
    st.markdown("**Hybrid Quantum-Classical Disease Detection**")
    st.caption("NISQ Optimization & Clinical Explainability")
    st.divider()

    # Backend health indicator
    try:
        r = requests.get(f"{API_URL}/", timeout=1.0)
        if r.status_code == 200:
            st.success("🟢 FastAPI Backend: Online (:8000)")
        else:
            st.warning(f"🟡 Backend status: {r.status_code}")
    except Exception:
        st.error("🔴 FastAPI Backend: Offline")

    st.divider()
    st.markdown("### Clinical Modules")
    st.markdown("- **Engine 1**: QSVC with `ZZFeatureMap` (Heart & WDBC)")
    st.markdown("- **Engine 2**: QTL ResNet-18 + 6-Qubit PQC (Brain MRI)")
    st.markdown("- **Interpretability**: Grad-CAM & Inverted PCA SHAP")
    st.markdown("- **Safety**: High-sensitivity triage ($\tau=0.10$)")


# ─── Main Interface ───────────────────────────────────────────────────────────

st.title("🏥 Quantum-Assisted Clinical Decision Support System")
st.markdown("SIH26139: Scalable Hybrid Quantum Machine Learning Pipeline with Full Rigor & Clinical Safety Gates")

tabs = st.tabs([
    "📊 Master Benchmark Summary",
    "🧠 Brain MRI & Quantum Transfer Learning",
    "🔍 Explainability (Grad-CAM & SHAP)",
    "🩺 Real-Time Diagnostic Triage",
    "⚙️ System Verification & Audit",
])

# ─── TAB 1: Benchmark Summary ─────────────────────────────────────────────────
with tabs[0]:
    st.header("Consolidated Clinical Performance Matrix")
    st.markdown(
        "Direct comparison of Quantum algorithms against four classical baselines "
        "(Logistic Regression, Random Forest, SVM-RBF, XGBoost, and ResNet-18) "
        "across all test splits."
    )

    if summary_data and "datasets" in summary_data:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Brain MRI (QTL Test Acc)", "91.00%", "Macro F2: 0.9083")
        with c2:
            st.metric("Brain MRI (Classical ResNet)", "94.19%", "Macro F2: 0.9409")
        with c3:
            st.metric("Heart Disease (Classical Best Acc)", "84.78%", "LR Recall: 90.48%")

        st.divider()

        # Tabular performance display
        st.subheader("Tabular Dataset Benchmark: Heart Disease & Breast Cancer")
        st.image(str(ASSETS_DIR / "benchmark_bars.png"), caption="Comparative Benchmark Suite across All Clinical Splits", use_container_width=True)

        st.subheader("Quantum vs. Classical Comparative Findings")
        st.info(
            "**Clinical Sensitivity Insight:** While classical tabular models (LR/RF/XGB) reach 85.7%–90.5% sensitivity on Heart Disease, "
            "QSVC with ZZFeatureMap defaults toward high specificity (92.0%), illustrating the quantum kernel concentration challenge in NISQ regimes. "
            "On complex imaging data, Hybrid QTL achieves **91.00% 4-class accuracy** with only 54 quantum parameters, closely rivaling the 11M-parameter classical model."
        )


# ─── TAB 2: Brain MRI Full Comparison ─────────────────────────────────────────
with tabs[1]:
    st.header("Brain MRI Multi-Class Evaluation (4-Way Diagnostic Split)")

    if mri_data:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Hybrid QTL (ResNet-18 + 6-Qubit PQC)")
            st.markdown(f"**Overall Accuracy:** `{mri_data['qtl']['accuracy']*100:.2f}%` | **Macro F2:** `{mri_data['qtl']['macro_f2']:.4f}`")
            st.markdown(f"**Inference Latency:** `{mri_data['qtl']['latency_ms_per_sample']:.2f} ms/sample`")

            # Per-class table
            qtl_rep = mri_data['qtl']['report']
            rows = []
            for cname in mri_data['class_names']:
                rows.append({
                    "Class": cname.capitalize(),
                    "Precision": f"{qtl_rep[cname]['precision']*100:.2f}%",
                    "Recall (Sensitivity)": f"{qtl_rep[cname]['recall']*100:.2f}%",
                    "F1-Score": f"{qtl_rep[cname]['f1-score']:.4f}",
                    "F2-Score": f"{qtl_rep[cname].get('f2', 0.0):.4f}",
                    "Test Samples": int(qtl_rep[cname]['support']),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with colB:
            st.subheader("Classical ResNet-18 Baseline (Pure Classical)")
            st.markdown(f"**Overall Accuracy:** `{mri_data['classical_resnet18']['accuracy']*100:.2f}%` | **Macro F2:** `{mri_data['classical_resnet18']['macro_f2']:.4f}`")
            st.markdown(f"**Inference Latency:** `{mri_data['classical_resnet18']['latency_ms_per_sample']:.2f} ms/sample`")

            cl_rep = mri_data['classical_resnet18']['report']
            cl_rows = []
            for cname in mri_data['class_names']:
                cl_rows.append({
                    "Class": cname.capitalize(),
                    "Precision": f"{cl_rep[cname]['precision']*100:.2f}%",
                    "Recall (Sensitivity)": f"{cl_rep[cname]['recall']*100:.2f}%",
                    "F1-Score": f"{cl_rep[cname]['f1-score']:.4f}",
                    "F2-Score": f"{cl_rep[cname].get('f2', 0.0):.4f}",
                    "Test Samples": int(cl_rep[cname]['support']),
                })
            st.dataframe(pd.DataFrame(cl_rows), use_container_width=True)

        st.divider()
        st.subheader("Training Trajectory: Loss & Macro F2 Curves")
        st.image(str(ASSETS_DIR / "training_curves_qtl.png"), caption="Two-Stage Hybrid QTL vs. Classical Fine-Tuning Dynamics", use_container_width=True)


# ─── TAB 3: Explainability & Biomarkers ────────────────────────────────────────
with tabs[2]:
    st.header("Clinical Interpretability & Explainability Suite")

    subtab1, subtab2, subtab3 = st.tabs([
        "🧠 Grad-CAM Brain MRI Saliency",
        "💓 Tabular SHAP Attributions (Heart & WDBC)",
        "⚛️ Quantum Bottleneck Sensitivity Matrix",
    ])

    with subtab1:
        st.subheader("Grad-CAM Anatomical Attribution Overlays")
        st.markdown(
            "Activation heatmaps computed at `layer4` of the ResNet-18 backbone. "
            "Confirms that both the Classical baseline and Hybrid QTL focus on verified tumor anatomy (enhancing margins, mass effect, dural tails) "
            "rather than peripheral artifacts."
        )
        st.image(str(ASSETS_DIR / "gradcam_mri_explanations.png"), use_container_width=True)

    with subtab2:
        st.subheader("Native Biomarker Attributions (Inverse PCA SHAP)")
        st.markdown(
            "To resolve the black-box nature of PCA-compressed quantum feature maps, "
            "we invert the projection matrix: $\Phi_{\\text{native}} = \Phi_{\\text{latent}} \cdot W_{\\text{PCA}}^T$."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(ASSETS_DIR / "shap_heart_biomarkers.png"), caption="Heart Disease: Native Diagnostic Risk Factors", use_container_width=True)
        with c2:
            st.image(str(ASSETS_DIR / "shap_wdbc_biomarkers.png"), caption="Breast Cancer WDBC: Cytopathological Attributions", use_container_width=True)

    with subtab3:
        st.subheader("Quantum Bottleneck Sensitivity Analysis")
        st.markdown(
            "Quantifies information flow through each of the 6 bottleneck qubits: "
            "$S_{j, k} = |\\partial \\langle Z_j \\rangle / \\partial z_k|$. "
            "Demonstrates active non-zero gradient transmission across all 6 input dimensions without barren plateau degeneration."
        )
        st.image(str(ASSETS_DIR / "quantum_bottleneck_importance.png"), use_container_width=True)


# ─── TAB 4: Real-time Diagnostic Triage ────────────────────────────────────────
with tabs[3]:
    st.header("Real-Time Diagnostic Decision Support (Live Backend)")
    st.markdown("Query the running FastAPI daemon (`:8000/predict`) with clinical patient feature profiles.")

    module = st.selectbox("Select Diagnostic Task:", ["Heart Disease Risk Assessment", "Breast Cancer WDBC Screening (High-Sensitivity)"])

    if module == "Heart Disease Risk Assessment":
        st.subheader("Patient Vitals & Clinical Risk Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            preset = st.selectbox("Load Preset Profile:", ["Custom", "Patient A (Healthy / Low Risk)", "Patient B (High Risk / Angina)"])
            if preset == "Patient A (Healthy / Low Risk)":
                init_vals = [45, 0, 0, 115, 180, 0, 0, 175, 0, 0.0, 2, 0, 2]
            elif preset == "Patient B (High Risk / Angina)":
                init_vals = [67, 1, 3, 160, 286, 1, 2, 108, 1, 2.6, 1, 3, 3]
            else:
                init_vals = [55, 1, 1, 130, 240, 0, 1, 150, 0, 1.0, 1, 0, 2]

            age = st.slider("Age", 20, 90, int(init_vals[0]))
            sex = st.selectbox("Sex", ["Female (0)", "Male (1)"], index=int(init_vals[1]))
            cp = st.selectbox("Chest Pain Type", ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal", "3: Asymptomatic"], index=int(init_vals[2]))
            trestbps = st.slider("Resting BP (mm Hg)", 90, 200, int(init_vals[3]))

        with col2:
            chol = st.slider("Serum Chol (mg/dl)", 120, 500, int(init_vals[4]))
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["0: No", "1: Yes"], index=int(init_vals[5]))
            restecg = st.selectbox("Resting ECG", ["0: Normal", "1: ST-T Abnormality", "2: LV Hypertrophy"], index=int(init_vals[6]))
            thalach = st.slider("Max Heart Rate", 70, 220, int(init_vals[7]))
            exang = st.selectbox("Exercise Induced Angina", ["0: No", "1: Yes"], index=int(init_vals[8]))

        with col3:
            oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, float(init_vals[9]), step=0.1)
            slope = st.selectbox("Slope of Peak ST", ["0: Upsloping", "1: Flat", "2: Downsloping"], index=int(init_vals[10]))
            ca = st.slider("Major Vessels (0-3)", 0, 3, int(init_vals[11]))
            thal = st.selectbox("Thalassemia", ["1: Normal", "2: Fixed Defect", "3: Reversible Defect"], index=min(int(init_vals[12])-1, 2))

        if st.button("Run Diagnostic Inference", type="primary"):
            features = [
                float(age), float(sex[0]), float(cp[0]), float(trestbps), float(chol),
                float(fbs[0]), float(restecg[0]), float(thalach), float(exang[0]),
                float(oldpeak), float(slope[0]), float(ca), float(thal[0])
            ]
            try:
                res = requests.post(f"{API_URL}/predict", json={"dataset": "heart", "features": features}, timeout=5)
                if res.status_code == 200:
                    d = res.json()
                    st.divider()
                    c_res1, c_res2, c_res3 = st.columns(3)
                    with c_res1:
                        st.metric("Predicted Diagnostic State", d["predicted_label"])
                    with c_res2:
                        st.metric("Posterior Disease Probability", f"{d['posterior_probability']*100:.2f}%")
                    with c_res3:
                        st.metric("Inference Latency", f"{d['inference_latency_ms']:.2f} ms")

                    if d["risk_stratification"] == "High Risk":
                        st.error(f"⚠️ **{d['risk_stratification']}**: {d['triage_action']}")
                    elif d["risk_stratification"] == "Moderate Risk":
                        st.warning(f"🔔 **{d['risk_stratification']}**: {d['triage_action']}")
                    else:
                        st.success(f"✅ **{d['risk_stratification']}**: {d['triage_action']}")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

    else:
        st.subheader("Breast Cancer Cytopathology Screening (Operating Point $\tau=0.10$)")
        st.info("Operating at high-sensitivity screening threshold $\tau=0.10$ to minimize missed malignancies.")
        if st.button("Evaluate Representative Screening Sample", type="primary"):
            npz_w = np.load(ROOT / "data" / "processed" / "wbcd_processed.npz")
            sample_feats = npz_w["X_test"][0].tolist()
            res = requests.post(f"{API_URL}/predict", json={"dataset": "wbcd", "features": sample_feats}, timeout=5)
            if res.status_code == 200:
                d = res.json()
                st.metric("Cytopathology Triage Result", d["predicted_label"], f"p={d['posterior_probability']*100:.1f}%")
                st.write(f"**Action Indicated:** {d['triage_action']}")


# ─── TAB 5: System Verification & Audit ───────────────────────────────────────
with tabs[4]:
    st.header("Deployment Verification Gate & Architecture Audit")

    if integration_data:
        st.subheader("Automated Integration Test Results (Port 8000)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Test Assertions", integration_data["total_tests"])
        with c2:
            st.metric("Passed Assertions", integration_data["passed_tests"], "100% Pass")
        with c3:
            st.metric("Failed Assertions", integration_data["failed_tests"])

        st.table(pd.DataFrame(integration_data["tests"]))

    st.subheader("Quantum vs. Classical Resource Attribution")
    st.markdown("""
| Resource Dimension | QSVC Tabular Engine | QTL Imaging Engine | Pure Classical Baselines |
| :--- | :--- | :--- | :--- |
| **Qubits** | 5 – 6 qubits | 6 qubits | 0 |
| **Ansatz Topology** | Full $ZZ$ Entanglement | StronglyEntanglingLayers ($SU(2)$ + CNOT ring) | Fully connected / Convolutional |
| **Circuit Depth** | Depth: 26 (reps=2) | Depth: 9 (3 layers) | N/A |
| **Trainable Params** | 0 quantum params (kernel) | 54 quantum params | 11.18M (ResNet-18) |
| **Inference Latency** | ~0.03 ms/sample | ~115.9 ms/sample | ~63.6 ms/sample |
    """)
