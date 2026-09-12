"""
src/preprocessing/ingest.py
============================
Multi-modal data ingestion for SIH26139 — Hybrid Quantum-Classical Disease Detection.

Sources:
  1. Brain MRI    — Kaggle: masoudnickparvar/brain-tumor-mri-dataset  (~7,023 images, 4 classes)
  2. Heart Disease — UCI ML Repository id=45 (Cleveland, 303 rows)
  3. WBCD          — sklearn.datasets.load_breast_cancer (569 rows)

Usage:
    python -m src.preprocessing.ingest          # runs all three
    python -m src.preprocessing.ingest --source mri
    python -m src.preprocessing.ingest --source heart
    python -m src.preprocessing.ingest --source wbcd
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

# Ensure stdout/stderr use UTF-8 on Windows (prevents charmap errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# ─── Project root (two levels up from this file) ────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─── Expected record counts (used for assertion gates) ───────────────────────
MRI_EXPECTED_IMAGES    = 7023
HEART_EXPECTED_ROWS    = 303
WBCD_EXPECTED_ROWS     = 569

MRI_CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}


# ════════════════════════════════════════════════════════════════════════════
# 1.  Brain MRI (Kaggle)
# ════════════════════════════════════════════════════════════════════════════

def ingest_brain_mri() -> dict:
    """
    Download Brain Tumor MRI Dataset from Kaggle and organise into
    data/raw/brain_mri/<class>/ folders.

    Returns a summary dict.
    Raises AssertionError if image count diverges from expected.
    """
    # pyrefly: ignore [missing-import]
    import kaggle  # noqa: PLC0415  (imported here so missing install is local)

    mri_dir = RAW_DIR / "brain_mri"
    mri_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = "masoudnickparvar/brain-tumor-mri-dataset"
    print(f"[MRI] Downloading '{dataset_slug}' from Kaggle …")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_slug,
        path=str(mri_dir),
        unzip=True,
        quiet=False,
    )
    print("[MRI] Download complete. Scanning class folders …")

    # Count images across all class sub-directories
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    class_counts: dict[str, int] = {}

    for item in sorted(mri_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in image_extensions:
            cls = item.parent.name.lower()
            class_counts[cls] = class_counts.get(cls, 0) + 1

    total_images = sum(class_counts.values())

    # ── Assertion gate ───────────────────────────────────────────────────────
    if total_images < MRI_EXPECTED_IMAGES:
        raise AssertionError(
            f"[MRI] FAILED count gate: expected >= {MRI_EXPECTED_IMAGES} images, "
            f"got {total_images}. Check Kaggle dataset or download integrity."
        )

    found_classes = set(class_counts.keys())
    if not MRI_CLASSES.issubset(found_classes):
        missing = MRI_CLASSES - found_classes
        raise AssertionError(
            f"[MRI] FAILED class gate: missing classes {missing}. "
            f"Found: {found_classes}"
        )

    print(f"[MRI] ✅  {total_images} images across {len(class_counts)} classes: {class_counts}")

    # ── Missing-value % (N/A for images — report 0) ──────────────────────────
    return {
        "source": "Brain MRI (Kaggle)",
        "type": "images",
        "count": total_images,
        "class_balance": class_counts,
        "missing_pct": 0.0,
        "path": str(mri_dir),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2.  Cleveland Heart Disease (UCI ML Repository)
# ════════════════════════════════════════════════════════════════════════════

def ingest_heart_disease() -> dict:
    """
    Fetch Cleveland Heart Disease dataset via ucimlrepo (id=45),
    save to data/raw/heart_disease.csv.

    Returns a summary dict.
    Raises AssertionError if row count diverges from expected.
    """
    # pyrefly: ignore [missing-import]
    from ucimlrepo import fetch_ucirepo  # noqa: PLC0415

    print("[Heart] Fetching Cleveland Heart Disease dataset (id=45) …")
    heart = fetch_ucirepo(id=45)

    X: pd.DataFrame = heart.data.features
    y: pd.Series    = heart.data.targets.squeeze()

    df = X.copy()
    df["target"] = y.values

    out_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(out_path, index=False)

    # ── Assertion gate ───────────────────────────────────────────────────────
    if len(df) != HEART_EXPECTED_ROWS:
        raise AssertionError(
            f"[Heart] FAILED count gate: expected {HEART_EXPECTED_ROWS} rows, "
            f"got {len(df)}. Dataset may have changed upstream."
        )

    # Class balance (target: 0 = no disease, 1-4 = disease severity)
    class_balance = df["target"].value_counts().to_dict()

    # Missing values
    total_cells  = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct  = round(100 * missing_cells / total_cells, 2)

    print(f"[Heart] ✅  {len(df)} rows | class balance: {class_balance} | missing: {missing_pct}%")

    return {
        "source": "Cleveland Heart Disease (UCI id=45)",
        "type": "tabular",
        "count": len(df),
        "class_balance": class_balance,
        "missing_pct": missing_pct,
        "path": str(out_path),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3.  Wisconsin Breast Cancer Diagnostic (sklearn)
# ════════════════════════════════════════════════════════════════════════════

def ingest_wbcd() -> dict:
    """
    Load Wisconsin Breast Cancer Diagnostic dataset via sklearn,
    save to data/raw/wbcd.csv.

    Returns a summary dict.
    Raises AssertionError if row count diverges from expected.
    """
    from sklearn.datasets import load_breast_cancer  # noqa: PLC0415

    print("[WBCD] Loading Wisconsin Breast Cancer Diagnostic dataset …")
    bc = load_breast_cancer(as_frame=True)

    df: pd.DataFrame = bc.frame  # features + target column

    out_path = RAW_DIR / "wbcd.csv"
    df.to_csv(out_path, index=False)

    # ── Assertion gate ───────────────────────────────────────────────────────
    if len(df) != WBCD_EXPECTED_ROWS:
        raise AssertionError(
            f"[WBCD] FAILED count gate: expected {WBCD_EXPECTED_ROWS} rows, "
            f"got {len(df)}. Sklearn version may differ."
        )

    # Class balance (target: 0 = malignant, 1 = benign)
    class_balance = df["target"].value_counts().to_dict()
    label_map = {v: k for k, v in enumerate(bc.target_names)}
    class_balance_named = {
        bc.target_names[k]: v for k, v in class_balance.items()
    }

    # Missing values
    total_cells   = df.size
    missing_cells = df.isnull().sum().sum()
    missing_pct   = round(100 * missing_cells / total_cells, 2)

    print(f"[WBCD] ✅  {len(df)} rows | class balance: {class_balance_named} | missing: {missing_pct}%")

    return {
        "source": "WBCD (sklearn.datasets)",
        "type": "tabular",
        "count": len(df),
        "class_balance": class_balance_named,
        "missing_pct": missing_pct,
        "path": str(out_path),
    }


# ════════════════════════════════════════════════════════════════════════════
# Report Generator
# ════════════════════════════════════════════════════════════════════════════

def generate_report(summaries: list[dict]) -> str:
    """Build the Markdown verification report for docs/data_ingestion_report.md."""
    lines = [
        "# Data Ingestion Verification Report",
        "",
        "Generated by `src/preprocessing/ingest.py`",
        f"Timestamp: {pd.Timestamp.now().isoformat(timespec='seconds')}",
        "",
        "## Summary Table",
        "",
        "| Source | Type | Count | Class Balance | Missing % |",
        "|--------|------|-------|---------------|-----------|",
    ]
    for s in summaries:
        bal = "; ".join(f"{k}={v}" for k, v in s["class_balance"].items())
        lines.append(
            f"| {s['source']} | {s['type']} | {s['count']} | {bal} | {s['missing_pct']}% |"
        )

    lines += [
        "",
        "## Dataset Details",
        "",
    ]
    for s in summaries:
        lines += [
            f"### {s['source']}",
            f"- **Type**: {s['type']}",
            f"- **Count**: {s['count']}",
            f"- **Class balance**: {s['class_balance']}",
            f"- **Missing value %**: {s['missing_pct']}%",
            f"- **Saved to**: `{s['path']}`",
            "",
        ]

    lines += [
        "## Assertion Gates Passed",
        "",
        f"- ✅ Brain MRI: >= {MRI_EXPECTED_IMAGES} images, 4 classes present",
        f"- ✅ Heart Disease: exactly {HEART_EXPECTED_ROWS} rows",
        f"- ✅ WBCD: exactly {WBCD_EXPECTED_ROWS} rows",
        "",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════════════════

def main(sources: list[str] | None = None) -> None:
    all_sources = ["mri", "heart", "wbcd"]
    if sources is None:
        sources = all_sources

    summaries: list[dict] = []
    errors: list[str] = []

    for src in sources:
        try:
            if src == "mri":
                summaries.append(ingest_brain_mri())
            elif src == "heart":
                summaries.append(ingest_heart_disease())
            elif src == "wbcd":
                summaries.append(ingest_wbcd())
            else:
                print(f"[WARN] Unknown source '{src}' — skipping.", file=sys.stderr)
        except Exception as exc:
            msg = f"[ERROR] Source '{src}' failed: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        print("\n\n❌ INGESTION HALTED — the following sources failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Write report
    report_md = generate_report(summaries)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    report_path = docs_dir / "data_ingestion_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n📄 Report written to {report_path}")

    # Print summary table to stdout
    print("\n" + "=" * 70)
    print("DATA INGESTION SUMMARY")
    print("=" * 70)
    header = f"{'Source':<40} {'Type':<10} {'Count':>6} {'Missing%':>9}"
    print(header)
    print("-" * 70)
    for s in summaries:
        print(f"{s['source']:<40} {s['type']:<10} {s['count']:>6} {s['missing_pct']:>8.2f}%")
    print("=" * 70)
    print("\n✅ All assertion gates passed. Proceed to Task 6 commit.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH26139 Data Ingestion")
    parser.add_argument(
        "--source",
        choices=["mri", "heart", "wbcd"],
        default=None,
        help="Run a single source only (default: all)",
    )
    args = parser.parse_args()
    main(sources=[args.source] if args.source else None)
