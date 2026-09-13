"""
src/classical/benchmark.py
============================
Central benchmark exporter for SIH26139.
Loads predictions from all trained models (quantum + classical), computes the
full metric suite on held-out test sets, runs McNemar's test between best
quantum and best classical per dataset, and exports:

  results/metrics/benchmark.json       — machine-readable, complete
  results/metrics/benchmark_report.md  — human-readable table
  results/figures/benchmark_bars.png   — comparison bar charts

Metrics computed per model x dataset:
  Accuracy, Precision, Recall/Sensitivity, Specificity, F1, F2 (beta=2),
  ROC-AUC (macro OvR), PR-AUC (macro OvR), MCC, McNemar p-value

Computational efficiency block:
  circuit_depth, gate_count, n_quantum_params, n_classical_params,
  training_time_s, inference_latency_ms

Usage
-----
    python -m src.classical.benchmark
    python -m src.classical.benchmark --verify-only  # just verify existing JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

METRICS_DIR    = ROOT / "results" / "metrics"
FIGURES_DIR    = ROOT / "results" / "figures"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
PROCESSED_DIR  = ROOT / "data" / "processed"

for _d in (METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe(val: Any) -> Any:
    """Convert numpy scalars to Python floats/ints for JSON serialisation."""
    if isinstance(val, (np.floating, float)):
        return round(float(val), 6) if not np.isnan(val) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    return val


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Specificity = TN / (TN + FP), macro-averaged across classes (OvR).
    For binary: straightforward; for multi-class: per-class OvR, then averaged.
    """
    specs = []
    for cls in range(n_classes):
        binary_true = (y_true == cls).astype(int)
        binary_pred = (y_pred == cls).astype(int)
        cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp = cm[0, 0], cm[0, 1]
            specs.append(tn / (tn + fp + 1e-9))
        else:
            specs.append(0.0)
    return float(np.mean(specs))


def _f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))


def _roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray | None, n_classes: int) -> float | None:
    if y_pred_proba is None:
        return None
    try:
        if n_classes == 2:
            return _safe(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            return _safe(roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray | None, n_classes: int) -> float | None:
    if y_pred_proba is None:
        return None
    try:
        if n_classes == 2:
            prec, rec, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            return _safe(auc(rec, prec))
        else:
            # Macro OvR PR-AUC
            pr_aucs = []
            for cls in range(n_classes):
                bin_true = (y_true == cls).astype(int)
                if bin_true.sum() == 0:
                    continue
                prec, rec, _ = precision_recall_curve(bin_true, y_pred_proba[:, cls])
                pr_aucs.append(auc(rec, prec))
            return _safe(np.mean(pr_aucs)) if pr_aucs else None
    except Exception:
        return None


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_classes: int = 2,
) -> dict:
    """Compute the full metric suite on one model's test predictions."""
    acc       = _safe(accuracy_score(y_true, y_pred))
    precision = _safe(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = _safe(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1        = _safe(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f2        = _safe(_f2_score(y_true, y_pred))
    mcc       = _safe(matthews_corrcoef(y_true, y_pred))
    spec      = _safe(_compute_specificity(y_true, y_pred, n_classes))
    roc_auc   = _roc_auc(y_true, y_proba, n_classes)
    pr_auc_v  = _pr_auc(y_true, y_proba, n_classes)

    return {
        "accuracy":     acc,
        "precision":    precision,
        "recall":       recall,
        "specificity":  spec,
        "f1":           f1,
        "f2":           f2,
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc_v,
        "mcc":          mcc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# McNemar's test
# ═══════════════════════════════════════════════════════════════════════════

def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """
    McNemar's chi-squared test (continuity corrected) between two classifiers.
    Tests H0: the two models make errors on the same samples.
    A low p-value (< 0.05) indicates a significant difference in error patterns.

    For multi-class: binary encode as (correct vs wrong) per sample, then apply test.

    Returns dict with chi2 statistic, p-value, and interpretation.
    """
    correct_a = (preds_a == y_true).astype(int)
    correct_b = (preds_b == y_true).astype(int)

    # Contingency: a_right_b_wrong, a_wrong_b_right
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))   # A correct, B wrong
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))   # A wrong, B correct

    # Continuity-corrected McNemar statistic
    if (b + c) == 0:
        chi2 = 0.0
        p    = 1.0
    else:
        chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
        p    = float(scipy_stats.chi2.sf(chi2, df=1))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "b": b,  # quantum right, classical wrong
        "c": c,  # quantum wrong, classical right
        "significant_at_05": p < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Computational efficiency — quantum circuits
# ═══════════════════════════════════════════════════════════════════════════

def _qsvc_efficiency(dataset: str) -> dict:
    """Extract QSVC circuit depth, gate count, timing from saved JSON."""
    result = {}
    for mode in ("noiseless", "noisy"):
        jpath = METRICS_DIR / f"qsvc_{dataset}_{mode}.json"
        if not jpath.exists():
            continue
        with open(jpath, encoding="utf-8-sig") as f:
            d = json.load(f)
        if d.get("status") not in ("complete", None):
            continue
        timing = d.get("timing", {})
        result[mode] = {
            "kernel_compute_s":       timing.get("kernel_compute_s"),
            "svm_fit_s":              timing.get("svm_fit_s"),
            "total_fit_s":            timing.get("total_fit_s"),
            "inference_latency_ms":   d.get("test", {}).get("latency_ms_per_sample"),
        }

    # Circuit metrics from Qiskit
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from src.preprocessing.tabular_pipeline import DATASET_CFG
        npz = np.load(PROCESSED_DIR / f"{dataset}_processed.npz")
        n_qubits = npz["X_train"].shape[1]
        fm = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="full")
        fm_decomp = fm.decompose()
        result["circuit_depth"]  = fm_decomp.depth()
        result["gate_count"]     = fm_decomp.size()
        result["n_qubits"]       = n_qubits
        result["n_quantum_params"] = 0          # QSVC: no trainable quantum params
        result["n_classical_params"] = "SVM dual (n_support_vectors)"
    except Exception as e:
        result["circuit_metrics_error"] = str(e)

    return result


def _qtl_efficiency() -> dict:
    """Extract QTL training time, circuit specs, param counts."""
    result = {}
    eff_path = METRICS_DIR / "qtl_efficiency.json"
    if eff_path.exists():
        with open(eff_path, encoding="utf-8") as f:
            result.update(json.load(f))

    # PennyLane circuit specs
    try:
        import pennylane as qml
        import numpy as np
        from models.qtl_architecture import _quantum_circuit, N_QUBITS, N_QNN_LAYERS
        dummy_inputs  = np.zeros(N_QUBITS)
        dummy_weights = np.zeros((N_QNN_LAYERS, N_QUBITS, 3))
        specs = qml.specs(_quantum_circuit)(dummy_inputs, dummy_weights)
        result["circuit_depth"]      = specs.get("depth", specs.get("num_depth", None))
        result["gate_count"]         = specs.get("num_gates", specs.get("num_operations", None))
        result["n_qubits"]           = N_QUBITS
        result["n_quantum_params"]   = N_QNN_LAYERS * N_QUBITS * 3  # 54
        result["n_classical_params"] = {
            "bottleneck": 512 * N_QUBITS + N_QUBITS,
            "output":     N_QUBITS * 4 + 4,   # output linear
            "backbone_trainable": "11M (frozen in S1, partial in S2)",
        }
    except Exception as e:
        result["circuit_metrics_error"] = str(e)

    # Inference latency from test run
    qtl_preds_path = METRICS_DIR / "qtl_test_preds.npy"
    if qtl_preds_path.exists():
        try:
            import torch
            from models.qtl_architecture import QuantumTransferModel
            model = QuantumTransferModel(freeze_backbone=True, pretrained=False)
            ckpt = CHECKPOINT_DIR / "qtl_best.pth"
            if not ckpt.exists():
                ckpt = CHECKPOINT_DIR / "qtl_stage1_best.pth"
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.eval()
            dummy = torch.randn(4, 3, 224, 224)
            # Warmup
            with torch.no_grad():
                _ = model(dummy)
            # Time over 20 samples
            t0 = time.perf_counter()
            reps = 5
            with torch.no_grad():
                for _ in range(reps):
                    _ = model(dummy)
            latency = (time.perf_counter() - t0) / (reps * 4) * 1000
            result["inference_latency_ms"] = round(latency, 2)
        except Exception as e:
            result["inference_latency_error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Load all predictions and build benchmark
# ═══════════════════════════════════════════════════════════════════════════

def _load_preds(path_stem: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load _preds.npy and _true.npy if they exist."""
    preds_path = METRICS_DIR / f"{path_stem}_preds.npy"
    true_path  = METRICS_DIR / f"{path_stem}_true.npy"
    if not preds_path.exists() or not true_path.exists():
        return None, None
    return np.load(preds_path), np.load(true_path)


def _load_proba(model_name: str, dataset: str) -> np.ndarray | None:
    """Try to load probability predictions from a fitted sklearn model."""
    ckpt = CHECKPOINT_DIR / f"classical_{model_name}_{dataset}.pkl"
    if not ckpt.exists():
        return None
    try:
        npz = np.load(PROCESSED_DIR / f"{dataset}_processed.npz")
        clf = joblib.load(ckpt)
        return clf.predict_proba(npz["X_test"])
    except Exception:
        return None


def build_benchmark() -> dict:
    """
    Load all saved predictions, compute metrics, run McNemar's tests.
    Returns the complete benchmark dict.
    """
    benchmark: dict[str, Any] = {
        "metadata": {
            "project": "SIH26139",
            "description": "Quantum vs Classical benchmark on disease detection datasets",
        },
        "tabular": {},
        "imaging": {},
        "computational_efficiency": {},
    }

    tabular_datasets  = ["heart", "wbcd"]
    classical_tabular = ["lr", "svm_rbf", "rf", "xgb"]

    # ────────────────────────────────────────────────────────────────────────
    # Tabular datasets
    # ────────────────────────────────────────────────────────────────────────
    for ds in tabular_datasets:
        benchmark["tabular"][ds] = {}
        npz = np.load(PROCESSED_DIR / f"{ds}_processed.npz")
        n_classes = len(np.unique(npz["y_test"]))

        # QSVC (noiseless)
        qsvc_preds, qsvc_true = _load_preds(f"qsvc_{ds}_noiseless")
        if qsvc_preds is not None:
            benchmark["tabular"][ds]["qsvc_noiseless"] = compute_full_metrics(
                qsvc_true, qsvc_preds, n_classes=n_classes
            )

        # QSVC (noisy)
        qsvc_noisy_preds, qsvc_noisy_true = _load_preds(f"qsvc_{ds}_noisy")
        if qsvc_noisy_preds is not None:
            benchmark["tabular"][ds]["qsvc_noisy"] = compute_full_metrics(
                qsvc_noisy_true, qsvc_noisy_preds, n_classes=n_classes
            )

        # Classical baselines
        for m in classical_tabular:
            preds, true = _load_preds(f"classical_{m}_{ds}")
            if preds is not None:
                proba = _load_proba(m, ds)
                benchmark["tabular"][ds][m] = compute_full_metrics(
                    true, preds, y_proba=proba, n_classes=n_classes
                )

        # McNemar's: QSVC noiseless vs best classical
        if qsvc_preds is not None:
            best_classical_name, best_classical_preds = None, None
            best_f2 = -1.0
            for m in classical_tabular:
                preds, true = _load_preds(f"classical_{m}_{ds}")
                if preds is not None:
                    f2 = _f2_score(true, preds)
                    if f2 > best_f2:
                        best_f2 = f2
                        best_classical_name  = m
                        best_classical_preds = preds
            if best_classical_preds is not None:
                mcn = mcnemar_test(qsvc_true, qsvc_preds, best_classical_preds)
                benchmark["tabular"][ds]["mcnemar"] = {
                    "quantum_model":   "qsvc_noiseless",
                    "classical_model": best_classical_name,
                    **mcn,
                }

        # Efficiency
        benchmark["computational_efficiency"][f"qsvc_{ds}"] = _qsvc_efficiency(ds)

    # ────────────────────────────────────────────────────────────────────────
    # Imaging dataset (Brain MRI)
    # ────────────────────────────────────────────────────────────────────────
    ds = "brain_mri"
    benchmark["imaging"][ds] = {}

    # QTL
    qtl_preds, qtl_true = _load_preds("qtl_test")
    if qtl_preds is not None:
        benchmark["imaging"][ds]["qtl"] = compute_full_metrics(
            qtl_true, qtl_preds, n_classes=4
        )

    # Classical ResNet-18
    cl_preds, cl_true = _load_preds("classical_resnet_test")
    if cl_preds is not None:
        benchmark["imaging"][ds]["classical_resnet18"] = compute_full_metrics(
            cl_true, cl_preds, n_classes=4
        )

    # McNemar's: QTL vs classical ResNet-18
    if qtl_preds is not None and cl_preds is not None:
        mcn = mcnemar_test(qtl_true, qtl_preds, cl_preds)
        benchmark["imaging"][ds]["mcnemar"] = {
            "quantum_model":   "qtl",
            "classical_model": "classical_resnet18",
            **mcn,
        }

    benchmark["computational_efficiency"]["qtl"] = _qtl_efficiency()

    return benchmark


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

def _render_markdown(bm: dict) -> str:
    lines = [
        "# SIH26139 Benchmark Report",
        "",
        "Generated by `src/classical/benchmark.py`",
        "",
        "## Tabular Datasets",
        "",
    ]
    METRIC_COLS = ["accuracy", "precision", "recall", "specificity",
                   "f1", "f2", "roc_auc", "pr_auc", "mcc"]

    for ds, models in bm.get("tabular", {}).items():
        lines += [f"### {ds.upper()}", ""]
        header = "| Model | " + " | ".join(c.replace("_", "\\_") for c in METRIC_COLS) + " |"
        sep    = "| --- |" + " --- |" * len(METRIC_COLS)
        lines += [header, sep]
        for model_name, metrics in models.items():
            if not isinstance(metrics, dict) or "accuracy" not in metrics:
                continue
            vals = [f"{metrics.get(c, 'N/A'):.4f}" if isinstance(metrics.get(c), float) else "N/A"
                    for c in METRIC_COLS]
            lines.append(f"| {model_name} | " + " | ".join(vals) + " |")

        # McNemar
        mcn = models.get("mcnemar", {})
        if mcn:
            lines += [
                "",
                f"**McNemar's test** ({mcn.get('quantum_model')} vs {mcn.get('classical_model')}):",
                f"- χ²={mcn.get('chi2', 'N/A')}  p={mcn.get('p_value', 'N/A')}  "
                f"significant={'Yes' if mcn.get('significant_at_05') else 'No'} (α=0.05)",
                "",
            ]

    lines += ["", "## Imaging Dataset (Brain MRI)", ""]
    for ds, models in bm.get("imaging", {}).items():
        lines += [f"### {ds}", ""]
        header = "| Model | " + " | ".join(c.replace("_", "\\_") for c in METRIC_COLS) + " |"
        sep    = "| --- |" + " --- |" * len(METRIC_COLS)
        lines += [header, sep]
        for model_name, metrics in models.items():
            if not isinstance(metrics, dict) or "accuracy" not in metrics:
                continue
            vals = [f"{metrics.get(c, 'N/A'):.4f}" if isinstance(metrics.get(c), float) else "N/A"
                    for c in METRIC_COLS]
            lines.append(f"| {model_name} | " + " | ".join(vals) + " |")

        mcn = models.get("mcnemar", {})
        if mcn:
            lines += [
                "",
                f"**McNemar's test** ({mcn.get('quantum_model')} vs {mcn.get('classical_model')}):",
                f"- χ²={mcn.get('chi2', 'N/A')}  p={mcn.get('p_value', 'N/A')}  "
                f"significant={'Yes' if mcn.get('significant_at_05') else 'No'} (α=0.05)",
                "",
            ]

    lines += ["", "## Computational Efficiency", "", "| Model | Circuit Depth | Gates | Qubits | Quantum Params | Inference Latency (ms) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for key, eff in bm.get("computational_efficiency", {}).items():
        depth = eff.get("circuit_depth", "N/A")
        gates = eff.get("gate_count", "N/A")
        qb    = eff.get("n_qubits", "N/A")
        qp    = eff.get("n_quantum_params", "N/A")
        lat   = eff.get("inference_latency_ms", "N/A")
        lines.append(f"| {key} | {depth} | {gates} | {qb} | {qp} | {lat} |")

    lines += ["", "---", "_SIH26139 — Quantum Disease Detection_"]
    return "\n".join(lines)


def _plot_bars(bm: dict, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [plot] matplotlib not available")
        return

    metrics_to_plot = ["f2", "roc_auc", "mcc", "accuracy"]
    datasets_info = []

    for ds, models in bm.get("tabular", {}).items():
        for m_name, m_vals in models.items():
            if isinstance(m_vals, dict) and "accuracy" in m_vals:
                datasets_info.append({"ds": ds, "model": m_name, **{k: m_vals.get(k) for k in metrics_to_plot}})

    for ds, models in bm.get("imaging", {}).items():
        for m_name, m_vals in models.items():
            if isinstance(m_vals, dict) and "accuracy" in m_vals:
                datasets_info.append({"ds": ds, "model": m_name, **{k: m_vals.get(k) for k in metrics_to_plot}})

    if not datasets_info:
        print("  [plot] No data to plot — skipping benchmark_bars.png")
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    fig.suptitle("Quantum vs Classical — Benchmark Comparison", fontsize=13, fontweight="bold")

    colors = {
        "qsvc_noiseless": "#1f77b4", "qsvc_noisy": "#aec7e8",
        "qtl": "#2ca02c",
        "lr": "#ff7f0e", "svm_rbf": "#d62728", "rf": "#9467bd",
        "xgb": "#8c564b", "classical_resnet18": "#e377c2",
    }

    for ax, metric in zip(axes if n_metrics > 1 else [axes], metrics_to_plot):
        labels = [f"{r['ds']}\n{r['model']}" for r in datasets_info]
        values = [r.get(metric) or 0.0 for r in datasets_info]
        bar_colors = [colors.get(r["model"], "#7f7f7f") for r in datasets_info]
        bars = ax.barh(labels, values, color=bar_colors, height=0.6)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(metric.upper())
        ax.set_title(metric.replace("_", " ").title())
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Benchmark bar chart saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Verification (Task 7 logic)
# ═══════════════════════════════════════════════════════════════════════════

def verify_benchmark(bm: dict) -> list[str]:
    """
    Check completeness of benchmark.json.
    Returns list of error strings (empty = all clear).
    """
    errors = []

    # Check tabular entries
    for ds in bm.get("tabular", {}):
        for model, metrics in bm["tabular"][ds].items():
            if model == "mcnemar":
                p = metrics.get("p_value")
                if p is None:
                    errors.append(f"tabular.{ds}.mcnemar.p_value is null")
                elif not (0.0 <= p <= 1.0):
                    errors.append(f"tabular.{ds}.mcnemar.p_value={p} out of [0,1]")
                continue
            for required in ["accuracy", "f2", "mcc"]:
                v = metrics.get(required)
                if v is None:
                    errors.append(f"tabular.{ds}.{model}.{required} is null")

    # Check imaging entries
    for ds in bm.get("imaging", {}):
        for model, metrics in bm["imaging"][ds].items():
            if model == "mcnemar":
                p = metrics.get("p_value")
                if p is None:
                    errors.append(f"imaging.{ds}.mcnemar.p_value is null")
                continue
            for required in ["accuracy", "f2", "mcc"]:
                v = metrics.get(required)
                if v is None:
                    errors.append(f"imaging.{ds}.{model}.{required} is null")

    # Check computational_efficiency is present
    if not bm.get("computational_efficiency"):
        errors.append("computational_efficiency block is missing or empty")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 Benchmark Exporter")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing benchmark.json without recomputing")
    args = parser.parse_args()

    bm_path = METRICS_DIR / "benchmark.json"

    if args.verify_only:
        if not bm_path.exists():
            print(f"[ERROR] {bm_path} not found. Run without --verify-only first.")
            sys.exit(1)
        with open(bm_path, encoding="utf-8") as f:
            bm = json.load(f)
        errors = verify_benchmark(bm)
        if errors:
            print("[FAIL] Benchmark verification failed:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("[OK] Benchmark verification PASSED — all entries present and non-null.")
        return

    print("\n[Benchmark] Building comprehensive metrics report...")
    bm = build_benchmark()

    # Save JSON
    with open(bm_path, "w", encoding="utf-8") as f:
        json.dump(bm, f, indent=2, default=str)
    print(f"[Benchmark] JSON saved to {bm_path}")

    # Save Markdown
    md = _render_markdown(bm)
    md_path = METRICS_DIR / "benchmark_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[Benchmark] Report saved to {md_path}")

    # Save bar chart
    _plot_bars(bm, FIGURES_DIR / "benchmark_bars.png")

    # Verify
    errors = verify_benchmark(bm)
    if errors:
        print("\n[WARNING] Benchmark has incomplete entries (some models may not be trained yet):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[OK] Benchmark verification PASSED — all entries complete.")


if __name__ == "__main__":
    main()
