"""
tests/test_quantum_engines.py
==============================
Task 8 -- Verification gate for both quantum inference engines.

Tests:
  1. QSVC import and circuit diagram generation
  2. Tabular pipeline: full preprocessing + QSVC noiseless sanity-check (toy data)
  3. QTL hybrid model: forward pass assertion output shape [batch, 4]
  4. Circuit diagrams saved to results/figures/
  5. QSVC results saved to results/metrics/

Run (from project root):
    $env:PYTHONUTF8="1"
    venv\\Scripts\\python.exe -X utf8 -m pytest tests/test_quantum_engines.py -v --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
PROCESSED_DIR = ROOT / "data" / "processed"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_toy_tabular(n_samples: int = 30, n_features: int = 6, n_classes: int = 2):
    """Generate tiny labelled dataset for QSVC smoke-test (no Kaggle/UCI needed)."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, np.pi, size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    # Ensure at least 1 sample per class in each split
    split = int(0.7 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — QSVC Tests
# ════════════════════════════════════════════════════════════════════════════

class TestQSVCEngine:

    def test_imports(self):
        """All Qiskit ML imports resolve without error."""
        from src.quantum.qsvc_engine import build_zz_feature_map, build_noise_model, build_qsvc
        assert True

    def test_zz_feature_map_shape(self):
        """ZZFeatureMap builds for n_qubits=6 with correct dimension."""
        from src.quantum.qsvc_engine import build_zz_feature_map
        fm = build_zz_feature_map(n_qubits=6, reps=2)
        assert fm.num_qubits == 6
        assert fm.feature_dimension == 6

    def test_zz_feature_map_shape_5qubits(self):
        """ZZFeatureMap builds for n_qubits=5."""
        from src.quantum.qsvc_engine import build_zz_feature_map
        fm = build_zz_feature_map(n_qubits=5, reps=2)
        assert fm.num_qubits == 5

    def test_circuit_diagram_saved(self):
        """ZZFeatureMap circuit diagram is written to results/figures/."""
        from src.quantum.qsvc_engine import save_circuit_diagram
        diagram = save_circuit_diagram(n_qubits=6, reps=2)
        assert isinstance(diagram, str)
        assert len(diagram) > 10
        out_path = FIGURES_DIR / "qsvc_circuit.txt"
        assert out_path.exists(), f"Circuit diagram not found at {out_path}"
        content = out_path.read_text(encoding="utf-8")
        assert "ZZFeatureMap" in content or "q_" in content or "┤" in content

    def test_noise_model_builds(self):
        """NoiseModel constructs without error."""
        from src.quantum.qsvc_engine import build_noise_model
        nm = build_noise_model()
        assert nm is not None

    def test_qsvc_noiseless_toy(self):
        """QSVC (noiseless) fits and predicts on toy 6-feature data."""
        from src.quantum.qsvc_engine import build_qsvc
        X_train, y_train, X_test, y_test = _make_toy_tabular(n_samples=20, n_features=6, n_classes=2)

        qsvc, fm = build_qsvc(n_qubits=6, noise_model=None, reps=1)
        qsvc.fit(X_train, y_train)
        preds = qsvc.predict(X_test)

        assert preds.shape == y_test.shape
        # Sanity: predictions are valid class labels
        assert set(np.unique(preds)).issubset({0, 1})

    def test_qsvc_result_saved_to_metrics(self):
        """
        Fast QSVC sanity check using a 50-sample subset of WBCD processed data.
        (Full 398-sample kernel matrix takes 15+ min on CPU — see test_qsvc_full_wbcd_slow.)
        If processed data doesn't exist yet, this test is skipped.
        """
        wbcd_npz = PROCESSED_DIR / "wbcd_processed.npz"
        if not wbcd_npz.exists():
            pytest.skip("WBCD processed data not found — run tabular pipeline first")

        from src.quantum.qsvc_engine import build_qsvc
        data = np.load(wbcd_npz)
        # Use 50 train / 20 test samples — fast kernel matrix (50x50)
        X_train, y_train = data["X_train"][:50], data["y_train"][:50]
        X_test,  y_test  = data["X_test"][:20],  data["y_test"][:20]
        n_qubits = X_train.shape[1]

        qsvc, _ = build_qsvc(n_qubits=n_qubits, noise_model=None, reps=1)
        qsvc.fit(X_train, y_train)
        preds = qsvc.predict(X_test)

        assert preds.shape == y_test.shape
        # Write a lightweight result JSON so downstream tests can check it exists
        result = {
            "dataset": "wbcd_subset50",
            "mode": "noiseless",
            "n_qubits": n_qubits,
            "test_accuracy": float((preds == y_test).mean()),
        }
        out_path = METRICS_DIR / "qsvc_wbcd_noiseless.json"
        # Only write lightweight stub if real full benchmark run is not already present
        write_stub = True
        if out_path.exists():
            try:
                with open(out_path, encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("status") == "complete":
                    write_stub = False
            except Exception:
                pass
        if write_stub:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        assert out_path.exists()
        assert 0.0 <= result["test_accuracy"] <= 1.0
        print(f"\n  QSVC subset accuracy: {result['test_accuracy']:.3f}")

    @pytest.mark.slow
    def test_qsvc_full_wbcd_slow(self):
        """
        Full WBCD training (398 samples, ~15 min on CPU).
        Skipped in normal CI — run explicitly with: pytest -m slow
        """
        wbcd_npz = PROCESSED_DIR / "wbcd_processed.npz"
        if not wbcd_npz.exists():
            pytest.skip("WBCD processed data not found")
        from src.quantum.qsvc_engine import train_and_evaluate
        results = train_and_evaluate("wbcd", mode="noiseless")
        assert "test" in results["noiseless"]
        assert 0.0 <= results["noiseless"]["test"]["accuracy"] <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — Tabular Pipeline Tests
# ════════════════════════════════════════════════════════════════════════════

class TestTabularPipeline:

    def test_imports(self):
        """Tabular pipeline imports resolve."""
        from src.preprocessing.tabular_pipeline import run_pipeline, _scale_to_pi
        assert True

    def test_scaling_assertion_passes(self):
        """_scale_to_pi correctly bounds data to [0, π] and assertion passes."""
        import math
        from src.preprocessing.tabular_pipeline import _scale_to_pi
        rng = np.random.default_rng(0)
        X = rng.uniform(-100, 100, size=(50, 10)).astype(float)
        X_scaled, scaler = _scale_to_pi(X)
        assert X_scaled.min() >= -1e-9
        assert X_scaled.max() <= math.pi + 1e-9

    def test_pipeline_heart(self):
        """Heart disease pipeline runs end-to-end if raw data available."""
        heart_csv = ROOT / "data" / "raw" / "heart_disease.csv"
        if not heart_csv.exists():
            pytest.skip("heart_disease.csv not found")
        from src.preprocessing.tabular_pipeline import run_pipeline
        summary = run_pipeline("heart")
        assert summary["k_pca"] in (5, 6)
        assert summary["train_size"] > 0

    def test_pipeline_wbcd(self):
        """WBCD pipeline runs end-to-end if raw data available."""
        wbcd_csv = ROOT / "data" / "raw" / "wbcd.csv"
        if not wbcd_csv.exists():
            pytest.skip("wbcd.csv not found")
        from src.preprocessing.tabular_pipeline import run_pipeline
        summary = run_pipeline("wbcd")
        assert summary["k_pca"] in (5, 6)
        assert summary["n_samples"] == 569


# ════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — QTL Model Tests
# ════════════════════════════════════════════════════════════════════════════

class TestQTLModel:

    def test_imports(self):
        """QTL model imports resolve."""
        from models.qtl_architecture import QuantumTransferModel, save_circuit_diagram
        assert True

    def test_forward_pass_shape_batch4(self):
        """
        CRITICAL assertion (Task 8):
        Forward pass with batch_size=4 dummy images → output shape MUST be [4, 4].
        """
        from models.qtl_architecture import QuantumTransferModel
        model = QuantumTransferModel(n_classes=4, freeze_backbone=True, pretrained=False)
        model.eval()

        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, 4), (
            f"[FAIL] Expected output shape ({batch_size}, 4), got {output.shape}"
        )

    def test_forward_pass_shape_batch1(self):
        """Forward pass also works for batch_size=1."""
        from models.qtl_architecture import QuantumTransferModel
        model = QuantumTransferModel(n_classes=4, freeze_backbone=True, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 4)

    def test_bottleneck_range(self):
        """Bottleneck output is bounded to [0, π]."""
        import math
        from models.qtl_architecture import Bottleneck
        bn = Bottleneck()
        x = torch.randn(10, 512) * 100  # extreme input
        out = bn(x)
        assert out.min().item() >= -1e-6, f"Bottleneck min {out.min().item():.4f} < 0"
        assert out.max().item() <= math.pi + 1e-6, f"Bottleneck max {out.max().item():.4f} > π"

    def test_backbone_frozen(self):
        """ResNet-18 backbone parameters are frozen when freeze_backbone=True."""
        from models.qtl_architecture import QuantumTransferModel
        model = QuantumTransferModel(freeze_backbone=True, pretrained=False)
        backbone_params = list(model.backbone.parameters())
        assert all(not p.requires_grad for p in backbone_params), \
            "Some backbone parameters are not frozen!"

    def test_trainable_params_count(self):
        """
        With frozen backbone, trainable parameter count should be:
        bottleneck:     512*6 + 6 = 3078
        quantum_layer:  3*6*3     = 54
        output_layer:   4*4 + 4   = 20
        Total: ~3152
        """
        from models.qtl_architecture import QuantumTransferModel
        model = QuantumTransferModel(freeze_backbone=True, pretrained=False)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Allow a range (TorchLayer may have slightly different count)
        assert 3000 <= n_trainable <= 4000, \
            f"Trainable params {n_trainable} outside expected range [3000, 4000]"

    def test_qtl_circuit_diagram_saved(self):
        """PennyLane circuit diagram is written to results/figures/."""
        from models.qtl_architecture import save_circuit_diagram
        diagram = save_circuit_diagram(out_dir=FIGURES_DIR)
        assert isinstance(diagram, str)
        assert len(diagram) > 10

        out_path = FIGURES_DIR / "qtl_circuit.txt"
        assert out_path.exists(), f"QTL circuit not found at {out_path}"
        content = out_path.read_text(encoding="utf-8")
        assert "AngleEmbedding" in content or "RY" in content or "CNOT" in content

    def test_unfreeze_backbone(self):
        """Unfreezing backbone makes last parametrized layer group trainable (skips avgpool)."""
        from models.qtl_architecture import QuantumTransferModel
        model = QuantumTransferModel(freeze_backbone=True, pretrained=False)
        # Before: all backbone params frozen
        assert all(not p.requires_grad for p in model.backbone.parameters())
        model.unfreeze_backbone(unfreeze_layers=1)
        # After: at least some backbone params trainable
        backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
        assert len(backbone_trainable) > 0, (
            "unfreeze_backbone(1) should unfreeze layer4 (has params); "
            "check that avgpool is being correctly skipped."
        )
        # Total trainable should have increased beyond the baseline (bottleneck+quantum+output)
        baseline = 3152  # bottleneck + quantum + output only
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable > baseline, (
            f"Total trainable {total_trainable} should exceed baseline {baseline} after unfreeze"
        )


# ════════════════════════════════════════════════════════════════════════════
# Integration — Run diagrams even without full pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestCircuitDiagrams:

    def test_both_diagrams_exist_after_tests(self):
        """After running other tests, both circuit diagram files should exist."""
        # Note: This test must run AFTER TestQSVCEngine and TestQTLModel
        qsvc_path = FIGURES_DIR / "qsvc_circuit.txt"
        qtl_path  = FIGURES_DIR / "qtl_circuit.txt"
        # Only check if prior tests have run; skip if files don't exist
        if not qsvc_path.exists():
            pytest.skip("QSVC circuit diagram not yet generated")
        if not qtl_path.exists():
            pytest.skip("QTL circuit diagram not yet generated")
        assert qsvc_path.stat().st_size > 0
        assert qtl_path.stat().st_size > 0
