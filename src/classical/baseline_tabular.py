"""
src/classical/baseline_tabular.py
===================================
Classical baseline models for tabular disease datasets (SIH26139).
Trains on the SAME stratified splits produced by tabular_pipeline.py.

Models trained per dataset (heart-disease + WBCD):
  1. Logistic Regression       (C=1.0, max_iter=1000, solver='lbfgs')
  2. SVM with RBF kernel       (C=10.0, gamma='scale')
  3. Random Forest             (n_estimators=300, random_state=42)
  4. XGBoost                   (n_estimators=300, max_depth=6, lr=0.1)

Outputs per model × dataset:
  models/checkpoints/classical_{model}_{dataset}.pkl   — fitted model
  results/metrics/classical_{model}_{dataset}.json     — accuracy + report
  results/metrics/classical_{model}_{dataset}_preds.npy — test predictions

Usage
-----
    python -m src.classical.baseline_tabular              # all models, both datasets
    python -m src.classical.baseline_tabular --dataset heart
    python -m src.classical.baseline_tabular --model lr wbcd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR    = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model registry ──────────────────────────────────────────────────────────
def _make_models() -> dict:
    return {
        "lr": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42
        ),
        "svm_rbf": SVC(
            C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        ),
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_baselines(dataset: str, model_keys: list[str] | None = None) -> dict[str, dict]:
    npz_path = PROCESSED_DIR / f"{dataset}_processed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run tabular_pipeline first."
        )
    d = np.load(npz_path)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val,   y_val   = d["X_val"],   d["y_val"]
    X_test,  y_test  = d["X_test"],  d["y_test"]

    print(f"\n{'='*60}")
    print(f"  CLASSICAL BASELINES: {dataset.upper()}")
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    print(f"{'='*60}")

    models_dict = _make_models()
    if model_keys:
        models_dict = {k: v for k, v in models_dict.items() if k in model_keys}

    all_results = {}

    for name, clf in models_dict.items():
        print(f"\n  [{name.upper()}]")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        def _eval(X: np.ndarray, y: np.ndarray, split: str) -> dict:
            t1 = time.perf_counter()
            preds = clf.predict(X)
            latency_ms = (time.perf_counter() - t1) / len(X) * 1000
            acc = float(accuracy_score(y, preds))
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            print(f"    {split:5s}  acc={acc:.4f}  latency={latency_ms:.3f}ms/sample")
            return {
                "accuracy": acc,
                "report": report,
                "latency_ms_per_sample": round(latency_ms, 4),
            }

        result = {
            "model": name,
            "dataset": dataset,
            "fit_time_s": round(fit_time, 4),
            "train": _eval(X_train, y_train, "train"),
            "val":   _eval(X_val,   y_val,   "val"),
            "test":  _eval(X_test,  y_test,  "test"),
        }

        # Save model
        ckpt = CHECKPOINT_DIR / f"classical_{name}_{dataset}.pkl"
        joblib.dump(clf, ckpt)

        # Save test predictions + ground truth
        test_preds = clf.predict(X_test)
        np.save(METRICS_DIR / f"classical_{name}_{dataset}_preds.npy", test_preds)
        np.save(METRICS_DIR / f"classical_{name}_{dataset}_true.npy",  y_test)

        # Save result JSON
        out_path = METRICS_DIR / f"classical_{name}_{dataset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"    fit={fit_time:.2f}s -> {out_path}")
        all_results[name] = result

    return all_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 Classical Tabular Baselines")
    parser.add_argument("--dataset", choices=["heart", "wbcd", "both"], default="both")
    parser.add_argument("--model", nargs="+",
                        choices=["lr", "svm_rbf", "rf", "xgb"],
                        default=None,
                        help="Models to train (default: all 4)")
    args = parser.parse_args()

    datasets = ["heart", "wbcd"] if args.dataset == "both" else [args.dataset]

    print("\n" + "="*60)
    print("CLASSICAL TABULAR BASELINES — SIH26139")
    print("="*60)

    for ds in datasets:
        results = train_baselines(ds, model_keys=args.model)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds in datasets:
        for m in (args.model or ["lr", "svm_rbf", "rf", "xgb"]):
            jpath = METRICS_DIR / f"classical_{m}_{ds}.json"
            if jpath.exists():
                with open(jpath) as f:
                    r = json.load(f)
                acc = r["test"]["accuracy"]
                print(f"  {ds:6s} {m:8s}  test_acc={acc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
