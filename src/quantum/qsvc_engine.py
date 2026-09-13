"""
src/quantum/qsvc_engine.py
===========================
Engine 1 — Quantum Support Vector Classifier (QSVC) for tabular disease datasets.
Targets: Cleveland Heart Disease and Wisconsin Breast Cancer (WBCD).

Architecture
------------
  ZZFeatureMap (Qiskit, 5–6 qubits)
    └─> FidelityQuantumKernel  K[i,j] = |<φ(x_i)|φ(x_j)>|²
        └─> QSVC (sklearn SVC wrapper with quantum kernel)

Barren Plateau Immunity — Design Guarantee
-------------------------------------------
This engine is provably immune to the barren plateau problem because:

1. The ZZFeatureMap is a FIXED, NON-PARAMETERISED feature map. Its gates
   (Ry, Rz rotations and ZZ entangling phases) encode classical data as angles
   but have NO trainable parameters of their own.

2. Optimization occurs ENTIRELY in the classical SVM dual problem:
       max  Σ_i α_i - (1/2) Σ_{i,j} α_i α_j y_i y_j K(x_i, x_j)
       s.t. 0 ≤ α_i ≤ C,  Σ_i α_i y_i = 0
   This is a STRICTLY CONVEX quadratic program solved by classical optimizers
   (e.g., SMO). It has a unique global minimum and requires NO gradient
   computation through the quantum circuit.

3. Barren plateaus are a pathology of GRADIENT-BASED optimization of deep
   parameterized quantum circuits (PQCs), where gradient magnitudes vanish
   exponentially with circuit depth/width. Since QSVC uses zero circuit
   gradients, this problem cannot arise by construction.

NISQ Noise Modeling (Task 3)
-----------------------------
Two separate run configurations are supported:
  (a) Noiseless: AerSimulator() with statevector method
  (b) Noisy: AerSimulator with NoiseModel combining:
        - Depolarizing error  (p1=0.001 single-qubit, p2=0.01 two-qubit)
        - Thermal relaxation  (T1=50μs, T2=70μs, gate_time=50ns)
      Calibrated to a representative IBM 7-qubit device noise profile.

Results saved to results/metrics/qsvc_noiseless.json and qsvc_noisy.json.

Usage
-----
    python -m src.quantum.qsvc_engine --dataset wbcd --mode both
    python -m src.quantum.qsvc_engine --dataset heart --mode noiseless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ── Qiskit imports ────────────────────────────────────────────────────────────
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
)
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[2]
PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR    = ROOT / "results" / "metrics"
FIGURES_DIR    = ROOT / "results" / "figures"

for _d in (METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Feature Map Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_zz_feature_map(n_qubits: int, reps: int = 2) -> ZZFeatureMap:
    """
    Build a ZZFeatureMap over n_qubits qubits with `reps` repetitions.

    Circuit structure (per rep):
      ┌───────────────────────┐
      │  Ry(x_i) ∀ i         │  ← angle embedding of data features
      │  Rz(x_i) ∀ i         │
      │  Rzz(2(π-x_i)(π-x_j))│  ← ZZ entangling phase gates (all pairs)
      └───────────────────────┘

    The ZZ interaction term implements the kernel:
        K(x, z) = |<0|U†(z) U(x)|0>|²
    where U = ZZFeatureMap circuit, capturing both first- and second-order
    feature correlations in Hilbert space.

    Args:
        n_qubits: Number of qubits (= dimensionality of PCA-compressed data).
        reps:     Number of circuit repetitions (depth). Default 2.

    Returns:
        Configured ZZFeatureMap circuit.
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement="full",   # all qubit pairs for maximum expressibility
    )
    return feature_map


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Noise Model (NISQ calibration — Task 3)
# ═══════════════════════════════════════════════════════════════════════════

def build_noise_model(
    p1: float = 0.001,   # single-qubit depolarizing probability
    p2: float = 0.01,    # two-qubit depolarizing probability
    t1: float = 50e3,    # T1 relaxation time (ns) — IBM device typical
    t2: float = 70e3,    # T2 dephasing time (ns)
    gate_time_1q: float = 50.0,   # single-qubit gate time (ns)
    gate_time_2q: float = 300.0,  # two-qubit gate time (ns)
) -> NoiseModel:
    """
    Construct a NISQ-representative noise model combining:

    Channel 1 — Depolarizing error
        Single-qubit: after every 1Q gate, apply depolarizing channel with prob p1.
        Two-qubit:    after every 2Q gate, apply depolarizing channel with prob p2.
        Mimics gate infidelity from control pulse imperfections.

    Channel 2 — Thermal relaxation (amplitude + phase damping)
        Models energy decay (T1) and dephasing (T2) during gate execution.
        Parameters calibrated to IBM 7-qubit device profile:
            T1 ~ 50μs, T2 ~ 70μs (published median values)

    Reference noise profile:
        IBM Nairobi (7-qubit) average T1=53μs, T2=68μs, CX gate error ~1%.

    Args:
        p1: Single-qubit depolarizing probability per gate.
        p2: Two-qubit depolarizing probability per gate.
        t1: T1 relaxation time in nanoseconds.
        t2: T2 coherence time in nanoseconds.
        gate_time_1q: Single-qubit gate duration in nanoseconds.
        gate_time_2q: Two-qubit gate duration in nanoseconds.

    Returns:
        Configured Qiskit Aer NoiseModel.
    """
    noise_model = NoiseModel()

    # ── Single-qubit gates ──────────────────────────────────────────────────
    # Depolarizing
    dep_err_1q = depolarizing_error(p1, num_qubits=1)
    # Thermal relaxation
    therm_err_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    # Compose: apply thermal first, then depolarizing
    combined_1q = therm_err_1q.compose(dep_err_1q)
    noise_model.add_all_qubit_quantum_error(
        combined_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "s", "sdg", "t", "tdg"]
    )

    # ── Two-qubit gates ─────────────────────────────────────────────────────
    dep_err_2q = depolarizing_error(p2, num_qubits=2)
    therm_err_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )
    combined_2q = therm_err_2q.compose(dep_err_2q)
    noise_model.add_all_qubit_quantum_error(combined_2q, ["cx", "cz", "rzz"])

    print(
        f"  NoiseModel built: p1={p1}, p2={p2}, "
        f"T1={t1/1e3:.0f}μs, T2={t2/1e3:.0f}μs, "
        f"gate_1q={gate_time_1q}ns, gate_2q={gate_time_2q}ns"
    )
    return noise_model


# ═══════════════════════════════════════════════════════════════════════════
# 3.  QSVC Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_qsvc(
    n_qubits: int,
    noise_model: NoiseModel | None = None,
    shots: int = 1024,
    reps: int = 2,
) -> tuple[QSVC, ZZFeatureMap]:
    """
    Build and return a configured QSVC classifier.

    Kernel computation:
        Uses StatevectorSampler (noiseless) or AerSampler with NoiseModel (noisy).
        FidelityQuantumKernel wraps ComputeUncompute fidelity estimation:
            K[i,j] = |<0|FM†(x_j) FM(x_i)|0>|²
        where FM = ZZFeatureMap.

    Args:
        n_qubits:    Dimensionality of input features (= PCA k_pca).
        noise_model: If provided, runs noisy simulation; else noiseless statevector.
        shots:       Number of measurement shots for noisy simulation.
        reps:        Feature map repetitions.

    Returns:
        (qsvc, feature_map) tuple.
    """
    feature_map = build_zz_feature_map(n_qubits, reps=reps)

    if noise_model is None:
        # Noiseless: use statevector sampler for exact kernel computation
        sampler = StatevectorSampler()
        label = "noiseless"
    else:
        # Noisy: use AerSampler with noise model via set_options (qiskit-aer 0.17.x API)
        from qiskit_aer.primitives import Sampler as AerSampler  # noqa: PLC0415
        sampler = AerSampler()
        sampler.set_options(noise_model=noise_model)
        label = "noisy"

    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
    )

    qsvc = QSVC(quantum_kernel=kernel)
    print(f"  QSVC built [{label}] — {n_qubits} qubits, {reps} reps, kernel: FidelityQuantumKernel")
    return qsvc, feature_map


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Training + Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    dataset_name: str,
    mode: Literal["noiseless", "noisy", "both"] = "both",
) -> dict:
    """
    Load preprocessed data, train QSVC in noiseless and/or noisy modes,
    evaluate, and save results to results/metrics/.

    Args:
        dataset_name: "heart" or "wbcd".
        mode: Which simulation mode(s) to run.

    Returns:
        dict with results for each mode.
    """
    # Load processed data
    npz_path = PROCESSED_DIR / f"{dataset_name}_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {npz_path}\n"
            "Run `python -m src.preprocessing.tabular_pipeline` first."
        )
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    n_qubits = X_train.shape[1]
    print(f"\n  Dataset: {dataset_name}  |  n_qubits={n_qubits}")
    print(f"  Train={len(y_train)}  Val={len(y_val)}  Test={len(y_test)}")

    noise_model = build_noise_model() if mode != "noiseless" else None
    modes_to_run = (
        ["noiseless", "noisy"] if mode == "both"
        else [mode]
    )

    all_results: dict[str, dict] = {}

    for run_mode in modes_to_run:
        print(f"\n  --- [{run_mode.upper()}] ---")
        nm = noise_model if run_mode == "noisy" else None
        qsvc, feature_map = build_qsvc(n_qubits, noise_model=nm)

        t0 = time.perf_counter()
        qsvc.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        def _eval(X: np.ndarray, y: np.ndarray, split: str) -> dict:
            preds = qsvc.predict(X)
            acc = accuracy_score(y, preds)
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            print(f"  {split} accuracy: {acc:.4f}")
            return {"accuracy": acc, "report": report}

        results = {
            "dataset": dataset_name,
            "mode": run_mode,
            "n_qubits": n_qubits,
            "train_time_s": round(train_time, 2),
            "train": _eval(X_train, y_train, "train"),
            "val":   _eval(X_val,   y_val,   "val"),
            "test":  _eval(X_test,  y_test,  "test"),
        }

        # Save to results/metrics/
        out_path = METRICS_DIR / f"qsvc_{dataset_name}_{run_mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {out_path}")

        # Save fitted model
        ckpt_path = CHECKPOINT_DIR / f"qsvc_{dataset_name}_{run_mode}.pkl"
        joblib.dump(qsvc, ckpt_path)
        print(f"  Model saved to {ckpt_path}")

        all_results[run_mode] = results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Circuit Diagram Export
# ═══════════════════════════════════════════════════════════════════════════

def save_circuit_diagram(n_qubits: int = 6, reps: int = 2) -> str:
    """
    Draw the ZZFeatureMap circuit and save to results/figures/qsvc_circuit.txt.
    Returns the diagram string.
    """
    feature_map = build_zz_feature_map(n_qubits, reps=reps)
    # Decompose to basis gates for full visualization
    circuit_decomposed = feature_map.decompose()
    diagram = circuit_decomposed.draw(output="text", fold=-1)
    diagram_str = str(diagram)

    out_path = FIGURES_DIR / "qsvc_circuit.txt"
    out_path.write_text(
        f"ZZFeatureMap — {n_qubits} qubits, {reps} reps\n"
        f"{'='*60}\n"
        f"{diagram_str}\n",
        encoding="utf-8",
    )
    print(f"  Circuit diagram saved to {out_path}")
    return diagram_str


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 QSVC Engine")
    parser.add_argument("--dataset", choices=["heart", "wbcd", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=["noiseless", "noisy", "both"],
        default="noiseless",
        help="Simulation mode (default: noiseless for speed)",
    )
    parser.add_argument("--diagram-only", action="store_true", help="Only save circuit diagram")
    args = parser.parse_args()

    if args.diagram_only:
        save_circuit_diagram()
        return

    datasets = ["heart", "wbcd"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}\n  QSVC ENGINE: {ds.upper()}\n{'='*60}")
        train_and_evaluate(ds, mode=args.mode)


if __name__ == "__main__":
    main()
