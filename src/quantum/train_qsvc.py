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
from sklearn.metrics import accuracy_score, classification_report, fbeta_score

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


def _check_kernel_diagonal(K_train: "np.ndarray", n_check: int = 10) -> float:
    """
    Standalone kernel diagonal sanity check.

    For a correctly normalised fidelity kernel, K(x_i, x_i) must equal 1.0
    (the inner product of a quantum state with itself is always 1).  Deviation
    indicates a normalisation bug in the feature map or sampler.

    Returns max_deviation from 1.0 across the checked samples.
    Prints PASS / WARN result for audit.
    """
    n = min(n_check, K_train.shape[0])
    diag = np.array([K_train[i, i] for i in range(n)])
    max_dev = float(np.abs(diag - 1.0).max())
    mean_val = float(diag.mean())

    status = "PASS" if max_dev < 0.05 else "WARN — possible normalisation bug"
    print(f"  [Kernel diagonal check]  n_checked={n}")
    print(f"    K(x,x) values: {diag.round(4).tolist()}")
    print(f"    mean={mean_val:.4f}  max_deviation_from_1={max_dev:.6f}  [{status}]")
    if max_dev >= 0.05:
        print(f"    *** Diagonal deviation {max_dev:.4f} ≥ 0.05 — inspect ZZFeatureMap "
              f"normalisation before trusting kernel results ***")
    return max_dev


def _val_c_sweep(
    K_train: "np.ndarray",
    y_train: "np.ndarray",
    K_val: "np.ndarray",
    y_val: "np.ndarray",
    c_values: list[float] | None = None,
) -> tuple[float, dict]:
    """
    Val-F2-score based C sweep on a precomputed kernel.

    Kernel matrices K_train and K_val are computed ONCE externally and reused
    for every C — no quantum re-evaluation per C value.

    Selection criterion: validation F2-score (beta=2, recall-weighted) to
    match the clinical objective and the selection metric used for classical
    baseline tuning in Priority 4.

    Returns (best_C, {str(c): val_f2, ...}).
    """
    from sklearn.svm import SVC

    if c_values is None:
        c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    results: dict[float, float] = {}
    best_c = c_values[0]
    best_val_f2 = -1.0
    for c in c_values:
        svm = SVC(kernel="precomputed", C=c)
        svm.fit(K_train, y_train)
        preds = svm.predict(K_val)
        f2 = float(fbeta_score(y_val, preds, beta=2, average="binary", zero_division=0))
        results[c] = round(f2, 4)
        if f2 > best_val_f2:
            best_val_f2 = f2
            best_c = c

    print(f"  C sweep (val F2):  " + "  ".join(f"C={c}: {results[c]:.3f}" for c in c_values))
    print(f"  Best C={best_c}  (val_F2={best_val_f2:.4f})")
    return best_c, results


def _train_single(
    dataset: str,
    mode: Literal["noiseless", "noisy"],
    data: dict[str, np.ndarray],
    reps: int = 2,
) -> dict:
    """
    Train one QSVC configuration. Returns result dict with:
      - kernel_compute_s: wall-clock for kernel matrix computation
      - svm_fit_s:        wall-clock for SVM dual solver (best-C model)
      - best_C:           C value selected by validation accuracy
      - val_acc_by_C:     C-sweep results for audit
      - train/val/test accuracy + classification report
    """
    from src.quantum.qsvc_engine import build_qsvc, build_noise_model
    import joblib
    from sklearn.svm import SVC

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    n_qubits = X_train.shape[1]

    print(f"\n  [{dataset.upper()} | {mode.upper()}]  n_qubits={n_qubits}  "
          f"train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    noise_model = build_noise_model() if mode == "noisy" else None
    qsvc, fm = build_qsvc(n_qubits=n_qubits, noise_model=noise_model, reps=reps)

    # ── Step 1: compute kernel matrices (timed) ────────────────────────────
    print(f"  Computing quantum kernel matrices...")
    t_kernel_start = time.perf_counter()
    K_train = qsvc.quantum_kernel.evaluate(X_train)            # (n_train, n_train)
    K_val   = qsvc.quantum_kernel.evaluate(X_val, X_train)     # (n_val,   n_train)
    K_test  = qsvc.quantum_kernel.evaluate(X_test, X_train)    # (n_test,  n_train)
    kernel_compute_s = time.perf_counter() - t_kernel_start

    # ── Kernel diagonal sanity check (standalone, before C sweep) ────────
    diag_dev = _check_kernel_diagonal(K_train)

    # ── Step 2: val-F2 based C sweep on precomputed kernel ────────────────
    print(f"  Running C sweep (selection by val F2-score, beta=2)...")
    best_c, val_f2_by_c = _val_c_sweep(K_train, y_train, K_val, y_val)

    # ── Step 3: final SVM fit with best C (timed separately) ─────────────
    t_svm_start = time.perf_counter()
    final_svm = SVC(kernel="precomputed", C=best_c)
    final_svm.fit(K_train, y_train)
    svm_fit_s = time.perf_counter() - t_svm_start
    total_fit_s = kernel_compute_s + svm_fit_s

    print(f"  kernel_compute_s={kernel_compute_s:.1f}  svm_fit_s={svm_fit_s:.3f}  "
          f"total={total_fit_s:.1f}s  best_C={best_c}")

    # ── Evaluation ──────────────────────────────────────────────────────────
    def _eval_precomputed(K: np.ndarray, y: np.ndarray, split: str) -> dict:
        t0 = time.perf_counter()
        preds = final_svm.predict(K)
        latency_ms = (time.perf_counter() - t0) / len(y) * 1000
        acc = float(accuracy_score(y, preds))
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        print(f"  {split} acc={acc:.4f}  latency={latency_ms:.2f}ms/sample")
        return {"accuracy": acc, "report": report, "latency_ms_per_sample": round(latency_ms, 3)}

    train_res = _eval_precomputed(K_train, y_train, "train")
    val_res   = _eval_precomputed(K_val,   y_val,   "val")
    test_res  = _eval_precomputed(K_test,  y_test,  "test")

    # Save test predictions for McNemar's test
    test_preds = final_svm.predict(K_test)
    np.save(METRICS_DIR / f"qsvc_{dataset}_{mode}_preds.npy", test_preds)
    np.save(METRICS_DIR / f"qsvc_{dataset}_{mode}_true.npy",  y_test)

    # Save the fitted SVM (not the full QSVC since kernel is precomputed separately)
    joblib.dump({"svm": final_svm, "qsvc": qsvc, "best_C": best_c},
                CHECKPOINT_DIR / f"qsvc_{dataset}_{mode}.pkl")

    # ── Kernel concentration diagnostic (WBCD-specific, NISQ failure mode) ─
    test_acc = test_res["accuracy"]
    concentration_note = None
    if dataset == "wbcd" and test_acc < 0.80:
        # Compute off-diagonal kernel statistics to diagnose concentration
        off_diag_vals = K_train[np.triu_indices(K_train.shape[0], k=1)]
        off_diag_mean = float(off_diag_vals.mean())
        off_diag_std  = float(off_diag_vals.std())
        concentration_note = (
            f"WBCD test accuracy {test_acc:.1%} < 80% target after C-sweep. "
            f"Off-diagonal kernel statistics: mean={off_diag_mean:.4f}, std={off_diag_std:.4f}. "
            f"Low std (< 0.05) indicates quantum kernel concentration (exponential "
            f"concentration / barren-plateau analogue in feature space) — a known "
            f"NISQ failure mode where deeply entangling ZZFeatureMaps produce kernel "
            f"matrices whose off-diagonal entries collapse toward a constant, removing "
            f"discriminative information. Mitigation: reduce reps, use linear entanglement, "
            f"or switch to data re-uploading PQC."
        )
        print(f"  [Kernel concentration] off-diag mean={off_diag_mean:.4f}  std={off_diag_std:.4f}")
        print(f"  {concentration_note}")

    result = {
        "dataset": dataset,
        "mode": mode,
        "status": "complete",
        "n_qubits": int(n_qubits),
        "reps": reps,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "best_C": best_c,
        "val_f2_by_C": {str(k): v for k, v in val_f2_by_c.items()},
        "kernel_diagonal_max_dev": round(diag_dev, 6),
        "kernel_concentration_note": concentration_note,
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
