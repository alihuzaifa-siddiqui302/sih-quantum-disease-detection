"""
src/preprocessing/tabular_pipeline.py
======================================
Preprocessing pipeline for tabular disease datasets in SIH26139.

Applies to:
  - Cleveland Heart Disease  (data/raw/heart_disease.csv,  303 rows, 14 cols, target col='target')
  - Wisconsin WBCD           (data/raw/wbcd.csv,           569 rows, 31 cols, target col='target')

Pipeline stages
---------------
1. Median imputation  (all numeric columns)
2. KNN imputation     (columns with structured missingness ≥ 1% NaN after median pass)
3. Min-Max scaling    strictly bounded to [0, π]  — asserted post-scaling
4. Feature selection  ANOVA F-test + mutual-information ranking → top 10–12 candidates
5. PCA                compress to k_pca components (default 6; falls back to 5 if 6th
                       component adds < 2% explained variance)
6. Stratified split   70 / 15 / 15  (train / val / test) — indices persisted as .npz
7. Artefact saving    fitted pipeline → models/checkpoints/<dataset>_tabular_pipeline.pkl

Usage
-----
    python -m src.preprocessing.tabular_pipeline --dataset heart
    python -m src.preprocessing.tabular_pipeline --dataset wbcd
    python -m src.preprocessing.tabular_pipeline  # runs both
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

for _d in (PROCESSED_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PI = math.pi
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)
RANDOM_STATE = 42

DATASET_CFG = {
    "heart": {
        "csv": RAW_DIR / "heart_disease.csv",
        "target_col": "target",
        "n_select": 12,     # ANOVA/MI candidates to shortlist
        "k_pca_max": 6,
        "binarize": True,   # collapse 0-4 severity to 0 (no disease) vs 1 (disease)
    },
    "wbcd": {
        "csv": RAW_DIR / "wbcd.csv",
        "target_col": "target",
        "n_select": 12,
        "k_pca_max": 6,
        "binarize": False,  # already binary (0=malignant, 1=benign)
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Stage helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Load CSV, separate features from target."""
    df = pd.read_csv(cfg["csv"])
    target_col = cfg["target_col"]
    y = df[target_col].values.astype(int)

    # Heart disease: binarize 0–4 severity → 0 (no disease) vs 1 (disease present)
    if cfg.get("binarize", False):
        y_before = np.unique(y).tolist()
        y = (y > 0).astype(int)
        print(f"  [BINARIZE] {target_col}: {y_before} → {{0: no-disease, 1: disease-present}}")
        print(f"  Class counts after binarize: 0={int((y==0).sum())}  1={int((y==1).sum())}")

    X = df.drop(columns=[target_col])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    print(f"  Loaded: {X.shape[0]} rows x {X.shape[1]} features  |  target unique: {np.unique(y)}")
    return X, y


def _impute(X: pd.DataFrame) -> np.ndarray:
    """
    Stage 1+2: median imputation first; then KNN imputation for any columns
    whose NaN rate was >= 1% in the original data (structured missingness).
    Returns numpy array.
    """
    missing_pct = X.isnull().mean()

    # Median pass (handles random / low-rate missingness)
    median_imp = SimpleImputer(strategy="median")
    X_median = median_imp.fit_transform(X)

    # Identify columns with original structured missingness >= 1%
    structured_cols = missing_pct[missing_pct >= 0.01].index.tolist()
    if structured_cols:
        col_indices = [X.columns.get_loc(c) for c in structured_cols]
        knn_imp = KNNImputer(n_neighbors=5)
        # Apply KNN only to affected columns, using median-imputed matrix as input
        X_knn = knn_imp.fit_transform(X_median)
        X_median[:, col_indices] = X_knn[:, col_indices]
        print(f"  KNN imputation applied to {len(structured_cols)} columns: {structured_cols}")
    else:
        print("  No structured missingness ≥1% — only median imputation used.")

    return X_median


def _scale_to_pi(X: np.ndarray) -> np.ndarray:
    """
    Stage 3: Min-Max scale to [0, π].
    Post-scaling assertion: min >= 0 and max <= π (within floating-point epsilon).
    """
    scaler = MinMaxScaler(feature_range=(0.0, PI))
    X_scaled = scaler.fit_transform(X)

    # ── Hard assertion gate ──────────────────────────────────────────────────
    _min = X_scaled.min()
    _max = X_scaled.max()
    assert _min >= -1e-9, f"[SCALE FAIL] min={_min:.6f} is below 0"
    assert _max <= PI + 1e-9, f"[SCALE FAIL] max={_max:.6f} exceeds π={PI:.6f}"
    print(f"  Scaling assertion passed: min={_min:.6f}  max={_max:.6f}  (bounds [0, π={PI:.4f}])")
    return X_scaled, scaler


def _select_features(
    X: np.ndarray,
    y: np.ndarray,
    n_select: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 4: Dual-criterion feature ranking.
    ANOVA F-test + mutual-information scores are averaged (rank-normalised)
    to pick top n_select features.
    Returns (X_selected, selected_indices).
    """
    n_features = X.shape[1]
    n_select = min(n_select, n_features)

    # ANOVA F-test scores (sensitive to linear relationships)
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # Mutual information scores (captures non-linear dependencies)
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # Rank-normalise to [0, 1] and average
    def _rank_norm(arr: np.ndarray) -> np.ndarray:
        ranks = arr.argsort().argsort().astype(float)
        return ranks / (len(ranks) - 1 + 1e-9)

    combined = (_rank_norm(f_scores) + _rank_norm(mi_scores)) / 2.0
    selected_idx = np.argsort(combined)[-n_select:][::-1]

    print(f"  Feature selection: {n_features} → {n_select} (top combined ANOVA+MI)")
    return X[:, selected_idx], selected_idx


def _apply_pca(
    X: np.ndarray,
    k_max: int = 6,
    min_variance_per_component: float = 0.02,
) -> tuple[np.ndarray, PCA, int]:
    """
    Stage 5: PCA compression.
    Tries k_max components; drops the last component if it contributes
    < min_variance_per_component of explained variance.
    Returns (X_pca, fitted_pca, k_final).
    """
    k_max = min(k_max, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=k_max, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {[f'{v:.3f}' for v in evr]}  (sum={evr.sum():.3f})")

    # Optionally drop last component if it contributes < threshold
    k_final = k_max
    if evr[-1] < min_variance_per_component and k_max > 5:
        k_final = k_max - 1
        X_pca = X_pca[:, :k_final]
        print(f"  Component {k_max} dropped (EVR={evr[-1]:.3f} < {min_variance_per_component}) → k={k_final}")
    else:
        print(f"  Using k={k_final} components (all above {min_variance_per_component} EVR threshold)")

    return X_pca, pca, k_final


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Stage 6: Stratified 70/15/15 split.
    Returns dict with keys 'train', 'val', 'test' each containing 'X' and 'y'.
    Persists indices to data/processed/<dataset>_splits.npz.
    """
    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE
    )
    train_idx, rest_idx = next(sss1.split(X, y))

    # Second split: val vs test from rest
    val_frac_of_rest = VAL_RATIO / (1 - TRAIN_RATIO)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_frac_of_rest), random_state=RANDOM_STATE
    )
    val_sub_idx, test_sub_idx = next(sss2.split(X[rest_idx], y[rest_idx]))
    val_idx  = rest_idx[val_sub_idx]
    test_idx = rest_idx[test_sub_idx]

    # Persist indices
    split_path = PROCESSED_DIR / f"{dataset_name}_splits.npz"
    np.savez(split_path, train=train_idx, val=val_idx, test=test_idx)
    print(f"  Split sizes — train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
    print(f"  Indices saved to {split_path}")

    return {
        "train": {"X": X[train_idx], "y": y[train_idx]},
        "val":   {"X": X[val_idx],   "y": y[val_idx]},
        "test":  {"X": X[test_idx],  "y": y[test_idx]},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(dataset_name: str) -> dict:
    """
    Execute the full tabular preprocessing pipeline for one dataset.
    Returns a summary dict.
    """
    cfg = DATASET_CFG[dataset_name]
    print(f"\n{'='*60}")
    print(f"  TABULAR PIPELINE: {dataset_name.upper()}")
    print(f"{'='*60}")

    # 1. Load
    print("\n[1] Loading data...")
    X_raw, y = _load(cfg)

    # 2. Impute
    print("\n[2] Imputation...")
    X_imp = _impute(X_raw)

    # 3. Scale to [0, π]
    print("\n[3] Scaling to [0, π]...")
    X_scaled, scaler = _scale_to_pi(X_imp)

    # 4. Feature selection
    print("\n[4] Feature selection...")
    X_sel, sel_idx = _select_features(X_scaled, y, cfg["n_select"])

    # 5. PCA
    print("\n[5] PCA dimensionality reduction...")
    X_pca, pca, k_final = _apply_pca(X_sel, k_max=cfg["k_pca_max"])

    # 6. Stratified split
    print("\n[6] Stratified train/val/test split...")
    splits = _stratified_split(X_pca, y, dataset_name)

    # 7. Save fitted objects
    print("\n[7] Saving fitted pipeline artefacts...")
    artefacts = {
        "scaler": scaler,
        "pca": pca,
        "selected_feature_indices": sel_idx,
        "k_pca": k_final,
        "dataset": dataset_name,
    }
    pkl_path = CHECKPOINT_DIR / f"{dataset_name}_tabular_pipeline.pkl"
    joblib.dump(artefacts, pkl_path)
    print(f"  Pipeline saved to {pkl_path}")

    # Save processed splits as npz (X+y per split)
    processed_path = PROCESSED_DIR / f"{dataset_name}_processed.npz"
    np.savez(
        processed_path,
        X_train=splits["train"]["X"], y_train=splits["train"]["y"],
        X_val=splits["val"]["X"],     y_val=splits["val"]["y"],
        X_test=splits["test"]["X"],   y_test=splits["test"]["y"],
    )
    print(f"  Processed data saved to {processed_path}")

    summary = {
        "dataset": dataset_name,
        "n_samples": X_pca.shape[0],
        "k_pca": k_final,
        "train_size": len(splits["train"]["y"]),
        "val_size":   len(splits["val"]["y"]),
        "test_size":  len(splits["test"]["y"]),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pipeline_pkl": str(pkl_path),
        "processed_npz": str(processed_path),
    }
    print(f"\n[OK] {dataset_name} pipeline complete. Output shape: {X_pca.shape}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 Tabular Preprocessing Pipeline")
    parser.add_argument(
        "--dataset",
        choices=["heart", "wbcd"],
        default=None,
        help="Run a single dataset (default: both)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["heart", "wbcd"]
    summaries = []
    for ds in datasets:
        s = run_pipeline(ds)
        summaries.append(s)

    print("\n\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    for s in summaries:
        print(
            f"  {s['dataset']:10s}  k={s['k_pca']}  "
            f"train={s['train_size']} val={s['val_size']} test={s['test_size']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
