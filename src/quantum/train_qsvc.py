"""
src/quantum/train_qsvc.py
==========================
Timed QSVC training on heart-disease and WBCD tabular datasets.
Records wall-clock kernel matrix computation time and SVM fit time separately.

Configurations run:
  heart  x noiseless   (~5-10 min)
  heart  x noisy       (~15-20 min)
  wbcd   x noiseless   (~15-30 min)
  wbcd   x noisy       (optional, 30+ min — writes "status":"DNF" if timeout)

Outputs per run:
  results/metrics/qsvc_{dataset}_{mode}.json     — full metrics + timing
  results/metrics/qsvc_{dataset}_{mode}_preds.npy — test predictions for McNemar's

Usage
-----
    python -m src.quantum.train_qsvc                          # all, noiseless only
    python -m src.quantum.train_qsvc --modes noiseless noisy  # all + noisy
    python -m src.quantum.train_qsvc --dataset heart --modes noiseless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR    = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

NOISY_WBCD_TIMEOUT_S = 2400   # 40 min — give WBCD noisy a generous timeout


# ═══════════════════════════════════════════════════════════════════════════
# Core training + timing
# ═══════════════════════════════════════════════════════════════════════════

def _load_split(dataset: str) -> dict[str, np.ndarray]:
    npz_path = PROCESSED_DIR / f"{dataset}_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {npz_path}\n"
            "Run `python -m src.preprocessing.tabular_pipeline` first."
        )
    d = np.load(npz_path)
    return {k: d[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}


def _train_single(
    dataset: str,
    mode: Literal["noiseless", "noisy"],
    data: dict[str, np.ndarray],
    reps: int = 2,
) -> dict:
    """
    Train one QSVC configuration. Returns result dict with:
      - kernel_compute_s: wall-clock for kernel matrix computation
      - svm_fit_s:        wall-clock for SVM dual solver
      - train/val/test accuracy + classification report
    """
    from src.quantum.qsvc_engine import build_qsvc, build_noise_model
    import joblib

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    n_qubits = X_train.shape[1]

    print(f"\n  [{dataset.upper()} | {mode.upper()}]  n_qubits={n_qubits}  "
          f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    noise_model = build_noise_model() if mode == "noisy" else None
    qsvc, fm = build_qsvc(n_qubits=n_qubits, noise_model=noise_model, reps=reps)

    # ── Kernel matrix computation (timed separately) ───────────────────────
    # Trigger kernel computation explicitly by fitting
    print(f"  Fitting QSVC (kernel + SVM combined timing)...")
    t_kernel_start = time.perf_counter()
    # The quantum kernel is computed during fit() — it's not separable from the
    # SVM solver in the Qiskit ML API without subclassing. We estimate:
    # kernel_compute_s = total_fit_s * (n_train^2 / (n_train^2 + n_train))  ≈ total_fit_s
    # and svm_fit_s from a classical SVM post-kernel (negligible vs kernel computation).
    qsvc.fit(X_train, y_train)
    t_fit_end = time.perf_counter()
    total_fit_s = t_fit_end - t_kernel_start

    # Classical SVM fit time estimation: fit a pre-computed kernel SVM
    from sklearn.svm import SVC
    import numpy as np
    K_train = qsvc.quantum_kernel.evaluate(X_train)
    t_svm_start = time.perf_counter()
    _svm_check = SVC(kernel="precomputed")
    _svm_check.fit(K_train, y_train)
    svm_fit_s = time.perf_counter() - t_svm_start
    kernel_compute_s = total_fit_s - svm_fit_s

    print(f"  kernel_compute_s={kernel_compute_s:.1f}  svm_fit_s={svm_fit_s:.3f}  total={total_fit_s:.1f}s")

    # ── Evaluation ──────────────────────────────────────────────────────────
    def _eval(X: np.ndarray, y: np.ndarray, split: str) -> dict:
        t0 = time.perf_counter()
        preds = qsvc.predict(X)
        latency_ms = (time.perf_counter() - t0) / len(X) * 1000
        acc = float(accuracy_score(y, preds))
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        print(f"  {split} acc={acc:.4f}  latency={latency_ms:.2f}ms/sample")
        return {"accuracy": acc, "report": report,
                "latency_ms_per_sample": round(latency_ms, 3)}

    train_res = _eval(X_train, y_train, "train")
    val_res   = _eval(X_val,   y_val,   "val")
    test_res  = _eval(X_test,  y_test,  "test")

    # Save test predictions for McNemar's test
    test_preds = qsvc.predict(X_test)
    preds_path = METRICS_DIR / f"qsvc_{dataset}_{mode}_preds.npy"
    true_path  = METRICS_DIR / f"qsvc_{dataset}_{mode}_true.npy"
    np.save(preds_path, test_preds)
    np.save(true_path, y_test)

    # Save model checkpoint
    ckpt_path = CHECKPOINT_DIR / f"qsvc_{dataset}_{mode}.pkl"
    import joblib
    joblib.dump(qsvc, ckpt_path)

    result = {
        "dataset": dataset,
        "mode": mode,
        "status": "complete",
        "n_qubits": int(n_qubits),
        "reps": reps,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "timing": {
            "total_fit_s": round(total_fit_s, 2),
            "kernel_compute_s": round(kernel_compute_s, 2),
            "svm_fit_s": round(svm_fit_s, 4),
        },
        "train": train_res,
        "val":   val_res,
        "test":  test_res,
    }

    out_path = METRICS_DIR / f"qsvc_{dataset}_{mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")
    return result


def _run_with_timeout(
    dataset: str,
    mode: Literal["noiseless", "noisy"],
    data: dict,
    timeout_s: float,
) -> dict:
    """
    Run _train_single with a soft timeout guard.
    If training takes longer than timeout_s, returns a DNF stub.
    """
    import threading
    result_container: list[dict] = []
    error_container:  list[str]  = []

    def _worker():
        try:
            result_container.append(_train_single(dataset, mode, data))
        except Exception as e:
            error_container.append(str(e))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        stub = {
            "dataset": dataset, "mode": mode, "status": "DNF",
            "reason": f"Exceeded timeout of {timeout_s/60:.0f} min on CPU",
        }
        out_path = METRICS_DIR / f"qsvc_{dataset}_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stub, f, indent=2)
        print(f"  [TIMEOUT] {dataset} {mode} — DNF after {timeout_s/60:.0f} min. Stub written.")
        return stub

    if error_container:
        stub = {"dataset": dataset, "mode": mode, "status": "ERROR", "reason": error_container[0]}
        print(f"  [ERROR] {dataset} {mode}: {error_container[0]}")
        return stub

    return result_container[0]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 QSVC Training")
    parser.add_argument("--dataset", choices=["heart", "wbcd", "both"], default="both")
    parser.add_argument("--modes", nargs="+",
                        choices=["noiseless", "noisy"],
                        default=["noiseless"],
                        help="Simulation modes to run (default: noiseless only)")
    parser.add_argument("--reps", type=int, default=2, help="ZZFeatureMap repetitions")
    args = parser.parse_args()

    datasets = ["heart", "wbcd"] if args.dataset == "both" else [args.dataset]
    all_results = {}

    for ds in datasets:
        data = _load_split(ds)
        all_results[ds] = {}

        for mode in args.modes:
            print(f"\n{'='*60}\n  QSVC: {ds.upper()} | {mode.upper()}\n{'='*60}")
            # Apply timeout only for noisy WBCD
            timeout = NOISY_WBCD_TIMEOUT_S if (ds == "wbcd" and mode == "noisy") else None
            if timeout:
                result = _run_with_timeout(ds, mode, data, timeout)
            else:
                result = _train_single(ds, mode, data, reps=args.reps)
            all_results[ds][mode] = result

    # Summary
    print("\n" + "="*60)
    print("QSVC TRAINING SUMMARY")
    print("="*60)
    for ds, modes_dict in all_results.items():
        for mode, r in modes_dict.items():
            status = r.get("status", "?")
            if status == "complete":
                acc = r.get("test", {}).get("accuracy", "N/A")
                t   = r.get("timing", {}).get("total_fit_s", "N/A")
                print(f"  {ds:6s} {mode:10s}  test_acc={acc:.4f}  fit={t:.0f}s")
            else:
                print(f"  {ds:6s} {mode:10s}  status={status}")
    print("="*60)


if __name__ == "__main__":
    main()
