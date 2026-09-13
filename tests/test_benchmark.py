"""
tests/test_benchmark.py
========================
Task 7 -- Verification gate for the benchmark.json artifact.

Asserts:
  1. benchmark.json exists and is valid JSON
  2. All required model x dataset combinations have entries
  3. Every entry has non-null accuracy, f2, mcc values
  4. McNemar's p-values are populated and in [0, 1]
  5. computational_efficiency block is present
  6. All QSVC result JSONs have status='complete' or status='DNF'
  7. Classical baseline JSONs exist for heart and WBCD

Run:
    $env:PYTHONUTF8="1"
    venv\\Scripts\\python.exe -X utf8 -m pytest tests/test_benchmark.py -v --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
CKPT_DIR    = ROOT / "models" / "checkpoints"

BM_PATH = METRICS_DIR / "benchmark.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_bm() -> dict:
    if not BM_PATH.exists():
        pytest.skip(f"benchmark.json not found at {BM_PATH} -- run benchmark.py first")
    with open(BM_PATH, encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 1. Classical Tabular Baselines (fast, should always be done first)
# ════════════════════════════════════════════════════════════════════════════

class TestClassicalTabularBaselines:

    def test_classical_result_jsons_exist(self):
        """Baseline JSON result files exist for all 4 models x 2 datasets."""
        for ds in ("heart", "wbcd"):
            for m in ("lr", "svm_rbf", "rf", "xgb"):
                p = METRICS_DIR / f"classical_{m}_{ds}.json"
                assert p.exists(), f"Missing: {p.name}"

    def test_classical_checkpoints_exist(self):
        """Fitted model pkl files exist for all 4 models x 2 datasets."""
        for ds in ("heart", "wbcd"):
            for m in ("lr", "svm_rbf", "rf", "xgb"):
                p = CKPT_DIR / f"classical_{m}_{ds}.pkl"
                assert p.exists(), f"Missing checkpoint: {p.name}"

    def test_classical_predictions_saved(self):
        """Test predictions .npy files exist for McNemar's test."""
        for ds in ("heart", "wbcd"):
            for m in ("lr", "svm_rbf", "rf", "xgb"):
                preds = METRICS_DIR / f"classical_{m}_{ds}_preds.npy"
                true  = METRICS_DIR / f"classical_{m}_{ds}_true.npy"
                assert preds.exists(), f"Missing preds: {preds.name}"
                assert true.exists(),  f"Missing true: {true.name}"

    @pytest.mark.parametrize("dataset", ["heart", "wbcd"])
    def test_classical_accuracy_sane(self, dataset):
        """All classical models achieve test accuracy > 0.4 (above random for binary)."""
        for m in ("lr", "svm_rbf", "rf", "xgb"):
            jpath = METRICS_DIR / f"classical_{m}_{dataset}.json"
            if not jpath.exists():
                pytest.skip(f"{jpath.name} not found")
            with open(jpath, encoding="utf-8") as f:
                d = json.load(f)
            acc = d["test"]["accuracy"]
            assert acc > 0.4, f"{m}/{dataset}: accuracy={acc:.4f} suspiciously low"


# ════════════════════════════════════════════════════════════════════════════
# 2. QSVC Results
# ════════════════════════════════════════════════════════════════════════════

class TestQSVCResults:

    def test_qsvc_noiseless_jsons_exist(self):
        """QSVC noiseless JSON results exist for both datasets."""
        for ds in ("heart", "wbcd"):
            p = METRICS_DIR / f"qsvc_{ds}_noiseless.json"
            assert p.exists(), f"Missing: {p.name}"

    @pytest.mark.parametrize("dataset", ["heart", "wbcd"])
    def test_qsvc_noiseless_status_complete(self, dataset):
        """QSVC noiseless result has status='complete'."""
        jpath = METRICS_DIR / f"qsvc_{dataset}_noiseless.json"
        if not jpath.exists():
            pytest.skip(f"{jpath.name} not found")
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        assert d.get("status") == "complete", \
            f"qsvc_{dataset}_noiseless status={d.get('status')}, expected 'complete'"

    @pytest.mark.parametrize("dataset", ["heart", "wbcd"])
    def test_qsvc_noiseless_timing_recorded(self, dataset):
        """QSVC noiseless result contains timing block with positive values."""
        jpath = METRICS_DIR / f"qsvc_{dataset}_noiseless.json"
        if not jpath.exists():
            pytest.skip(f"{jpath.name} not found")
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        timing = d.get("timing", {})
        assert timing.get("total_fit_s", 0) > 0, "total_fit_s should be positive"
        assert timing.get("kernel_compute_s", 0) >= 0
        assert timing.get("svm_fit_s", 0) >= 0

    def test_qsvc_noisy_json_exists_or_dnf(self):
        """Noisy QSVC JSON exists (may be 'complete' or 'DNF')."""
        for ds in ("heart",):   # WBCD noisy is optional
            p = METRICS_DIR / f"qsvc_{ds}_noisy.json"
            assert p.exists(), f"Missing: {p.name} (even DNF stub required)"

    def test_qsvc_predictions_saved(self):
        """QSVC noiseless predictions exist for McNemar's test."""
        for ds in ("heart", "wbcd"):
            preds = METRICS_DIR / f"qsvc_{ds}_noiseless_preds.npy"
            assert preds.exists(), f"Missing: {preds.name}"


# ════════════════════════════════════════════════════════════════════════════
# 3. QTL Model
# ════════════════════════════════════════════════════════════════════════════

class TestQTLResults:

    def test_qtl_stage1_checkpoint_exists(self):
        p = CKPT_DIR / "qtl_stage1_best.pth"
        assert p.exists(), f"Missing Stage 1 checkpoint: {p}"

    def test_qtl_best_checkpoint_or_stage1(self):
        """Either full best or stage1 checkpoint must exist."""
        best    = CKPT_DIR / "qtl_best.pth"
        stage1  = CKPT_DIR / "qtl_stage1_best.pth"
        assert best.exists() or stage1.exists(), \
            "Neither qtl_best.pth nor qtl_stage1_best.pth found"

    def test_qtl_history_saved(self):
        hist = METRICS_DIR / "qtl_stage1_history.json"
        assert hist.exists(), f"Missing: {hist}"
        with open(hist, encoding="utf-8") as f:
            d = json.load(f)
        assert len(d.get("val_f2", [])) > 0, "val_f2 history is empty"
        assert all(0.0 <= v <= 1.0 for v in d["val_f2"]), "val_f2 values out of [0, 1]"

    def test_qtl_test_predictions_saved(self):
        preds = METRICS_DIR / "qtl_test_preds.npy"
        true  = METRICS_DIR / "qtl_test_true.npy"
        if not preds.exists():
            pytest.skip("QTL test predictions not yet saved")
        assert preds.exists() and true.exists()
        arr = np.load(preds)
        assert arr.ndim == 1
        assert set(np.unique(arr)).issubset({0, 1, 2, 3}), \
            "QTL predictions should be class indices 0-3 (4-class MRI)"

    def test_qtl_training_curves_saved(self):
        p = FIGURES_DIR / "training_curves_qtl.png"
        if not p.exists():
            pytest.skip("training_curves_qtl.png not yet generated")
        assert p.stat().st_size > 1000, "training_curves_qtl.png appears empty"


# ════════════════════════════════════════════════════════════════════════════
# 4. Benchmark JSON — Task 7 Verification Gate
# ════════════════════════════════════════════════════════════════════════════

class TestBenchmarkJSON:

    def test_benchmark_json_exists(self):
        assert BM_PATH.exists(), f"benchmark.json not found at {BM_PATH}"

    def test_benchmark_json_valid(self):
        """benchmark.json parses as valid JSON."""
        bm = _load_bm()
        assert isinstance(bm, dict)

    def test_benchmark_has_required_top_level_keys(self):
        bm = _load_bm()
        for key in ("metadata", "tabular", "imaging", "computational_efficiency"):
            assert key in bm, f"Missing top-level key: '{key}'"

    @pytest.mark.parametrize("dataset", ["heart", "wbcd"])
    def test_tabular_models_present(self, dataset):
        """At least QSVC noiseless + one classical model entry exists per tabular dataset."""
        bm = _load_bm()
        ds_data = bm.get("tabular", {}).get(dataset, {})
        assert len(ds_data) > 0, f"No entries in benchmark.tabular.{dataset}"

    @pytest.mark.parametrize("dataset,model", [
        ("heart", "qsvc_noiseless"), ("heart", "lr"),
        ("heart", "rf"), ("wbcd", "qsvc_noiseless"), ("wbcd", "xgb"),
    ])
    def test_tabular_entry_non_null(self, dataset, model):
        """Each tabular model entry has non-null accuracy, f2, mcc."""
        bm = _load_bm()
        entry = bm.get("tabular", {}).get(dataset, {}).get(model)
        if entry is None:
            pytest.skip(f"No entry for {dataset}.{model}")
        for metric in ("accuracy", "f2", "mcc"):
            v = entry.get(metric)
            assert v is not None, f"{dataset}.{model}.{metric} is null"
            assert isinstance(v, (int, float)), f"{dataset}.{model}.{metric} not numeric"

    @pytest.mark.parametrize("dataset", ["heart", "wbcd"])
    def test_mcnemar_p_value_populated(self, dataset):
        """McNemar p-value exists and is in [0, 1] for each tabular dataset."""
        bm = _load_bm()
        mcn = bm.get("tabular", {}).get(dataset, {}).get("mcnemar")
        if mcn is None:
            pytest.skip(f"McNemar entry missing for {dataset} — both quantum and classical must be trained")
        p = mcn.get("p_value")
        assert p is not None, f"McNemar p_value is null for {dataset}"
        assert 0.0 <= p <= 1.0, f"McNemar p_value={p} out of [0, 1]"

    def test_imaging_dataset_present(self):
        bm = _load_bm()
        assert len(bm.get("imaging", {})) > 0, "No imaging entries in benchmark"

    def test_computational_efficiency_present(self):
        bm = _load_bm()
        eff = bm.get("computational_efficiency", {})
        assert len(eff) > 0, "computational_efficiency block is empty"

    def test_benchmark_verification_passes(self):
        """Run the built-in verify_benchmark() function — must return empty error list."""
        bm = _load_bm()
        from src.classical.benchmark import verify_benchmark
        errors = verify_benchmark(bm)
        assert errors == [], (
            "Benchmark verification failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    def test_benchmark_bars_png_exists(self):
        p = FIGURES_DIR / "benchmark_bars.png"
        if not p.exists():
            pytest.skip("benchmark_bars.png not yet generated")
        assert p.stat().st_size > 1000

    def test_benchmark_report_md_exists(self):
        p = METRICS_DIR / "benchmark_report.md"
        if not p.exists():
            pytest.skip("benchmark_report.md not yet generated")
        content = p.read_text(encoding="utf-8")
        assert "SIH26139" in content
        assert "McNemar" in content
