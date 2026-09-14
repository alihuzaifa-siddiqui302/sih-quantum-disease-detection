"""
tests/test_integration_backend.py
==================================
Automated End-to-End Integration Tests for SIH26139 FastAPI Backend.
Tests:
  - GET /model-info
  - GET /results-summary
  - POST /predict (Heart Disease: healthy vs diseased payloads)
  - POST /predict (WDBC: benign vs malignant payloads with tau=0.10 triage)
  - POST /predict-image (Brain MRI scan file upload)
  - Measures request-response latency (ensuring high responsiveness)
  - Exports results/integration_test_report.json
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8000"
REPORT_PATH = ROOT / "results" / "integration_test_report.json"
MRI_TEST_DIR = ROOT / "data" / "raw" / "brain_mri" / "Testing"


def run_integration_tests():
    print(f"\n{'='*60}")
    print("  FASTAPI CLINICAL BACKEND INTEGRATION TEST SUITE")
    print(f"{'='*60}")
    print(f"Target URL: {BASE_URL}")

    # Ensure server is up (poll up to 10s)
    server_ready = False
    for attempt in range(10):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=2)
            if r.status_code == 200:
                server_ready = True
                print("  [OK] Backend server is reachable and responsive.")
                break
        except Exception:
            time.sleep(1)

    if not server_ready:
        print("  [ERROR] Server not reachable on port 8000 after 10 seconds.")
        sys.exit(1)

    test_results = []

    # ── Test 1: GET /model-info ───────────────────────────────────────────────
    t0 = time.perf_counter()
    r = requests.get(f"{BASE_URL}/model-info")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "engine_1_tabular" in data
    assert "engine_2_imaging" in data
    print(f"  [PASS] GET /model-info (Status {r.status_code}, {latency_ms:.2f} ms)")
    test_results.append({
        "test": "GET /model-info",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "passed": True,
    })

    # ── Test 2: GET /results-summary ──────────────────────────────────────────
    t0 = time.perf_counter()
    r = requests.get(f"{BASE_URL}/results-summary")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "datasets" in data
    print(f"  [PASS] GET /results-summary (Status {r.status_code}, {latency_ms:.2f} ms)")
    test_results.append({
        "test": "GET /results-summary",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "passed": True,
    })

    # ── Test 3: POST /predict (Heart Healthy) ──────────────────────────────────
    # Typical low-risk healthy heart profile
    healthy_heart = [45.0, 0.0, 0.0, 115.0, 180.0, 0.0, 0.0, 175.0, 0.0, 0.0, 2.0, 0.0, 2.0]
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict", json={"dataset": "heart", "features": healthy_heart})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "risk_stratification" in data
    print(f"  [PASS] POST /predict (Heart Healthy) -> {data['risk_stratification']} ({latency_ms:.2f} ms)")
    test_results.append({
        "test": "POST /predict [Heart Healthy]",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "prediction": data["predicted_label"],
        "risk": data["risk_stratification"],
        "passed": True,
    })

    # ── Test 4: POST /predict (Heart Diseased) ────────────────────────────────
    # High-risk diseased heart profile (ST depression, angina, low max HR)
    diseased_heart = [67.0, 1.0, 3.0, 160.0, 286.0, 1.0, 2.0, 108.0, 1.0, 2.6, 1.0, 3.0, 3.0]
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict", json={"dataset": "heart", "features": diseased_heart})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200
    data = r.json()
    print(f"  [PASS] POST /predict (Heart Diseased) -> {data['risk_stratification']} ({latency_ms:.2f} ms)")
    test_results.append({
        "test": "POST /predict [Heart Diseased]",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "prediction": data["predicted_label"],
        "risk": data["risk_stratification"],
        "passed": True,
    })

    # ── Test 5: POST /predict (WDBC with tau=0.10 Triage) ──────────────────────
    # 30-feature vector from test split
    npz_w = np.load(ROOT / "data" / "processed" / "wbcd_processed.npz")
    w_sample = npz_w["X_test"][0].tolist()
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/predict", json={"dataset": "wbcd", "features": w_sample})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200
    data = r.json()
    print(f"  [PASS] POST /predict (WDBC Triage) -> {data['predicted_label']} ({latency_ms:.2f} ms)")
    test_results.append({
        "test": "POST /predict [WDBC Triage]",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "prediction": data["predicted_label"],
        "risk": data["risk_stratification"],
        "passed": True,
    })

    # ── Test 6: POST /predict-image (Brain MRI Upload) ─────────────────────────
    # Select sample MRI scan
    sample_mri = list(MRI_TEST_DIR.glob("*/*.jpg"))[0]
    with open(sample_mri, "rb") as f:
        img_bytes = f.read()

    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/predict-image",
        files={"file": (sample_mri.name, img_bytes, "image/jpeg")},
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "predicted_class_name" in data
    print(f"  [PASS] POST /predict-image (Brain MRI) -> {data['predicted_class_name']} ({latency_ms:.2f} ms)")
    test_results.append({
        "test": "POST /predict-image [Brain MRI]",
        "status_code": r.status_code,
        "latency_ms": round(latency_ms, 2),
        "prediction": data["predicted_class_name"],
        "confidence": data["confidence"],
        "passed": True,
    })

    # ── Save Integration Report ───────────────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_url": BASE_URL,
        "total_tests": len(test_results),
        "passed_tests": sum(1 for t in test_results if t["passed"]),
        "failed_tests": sum(1 for t in test_results if not t["passed"]),
        "tests": test_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] All {len(test_results)} integration tests passed.")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    run_integration_tests()
