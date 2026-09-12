"""
tests/test_scaffold.py
======================
Smoke tests to verify the project scaffold is intact.
"""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Directory presence ────────────────────────────────────────────────────────

def test_required_directories_exist():
    required = [
        "data/raw",
        "data/processed",
        "src/preprocessing",
        "src/quantum",
        "src/classical",
        "src/explainability",
        "src/dashboard",
        "src/api",
        "tests",
        "notebooks",
        "models/checkpoints",
        "results/metrics",
        "results/figures",
        "docs",
        ".github/workflows",
    ]
    for d in required:
        assert (ROOT / d).is_dir(), f"Missing directory: {d}"


# ── __init__.py presence ──────────────────────────────────────────────────────

def test_init_files_exist():
    pkgs = [
        "src/__init__.py",
        "src/preprocessing/__init__.py",
        "src/quantum/__init__.py",
        "src/classical/__init__.py",
        "src/explainability/__init__.py",
        "src/dashboard/__init__.py",
        "src/api/__init__.py",
    ]
    for f in pkgs:
        assert (ROOT / f).is_file(), f"Missing __init__.py: {f}"


# ── Key files ─────────────────────────────────────────────────────────────────

def test_requirements_txt_exists():
    assert (ROOT / "requirements.txt").is_file()


def test_gitignore_exists():
    assert (ROOT / ".gitignore").is_file()


def test_ingest_py_exists():
    assert (ROOT / "src" / "preprocessing" / "ingest.py").is_file()


# ── .gitignore gates ──────────────────────────────────────────────────────────

def test_gitignore_blocks_data():
    gi = (ROOT / ".gitignore").read_text()
    assert "data/" in gi, ".gitignore must block data/ directory"


def test_gitignore_blocks_venv():
    gi = (ROOT / ".gitignore").read_text()
    assert "venv/" in gi, ".gitignore must block venv/"


def test_gitignore_blocks_env():
    gi = (ROOT / ".gitignore").read_text()
    assert ".env" in gi, ".gitignore must block .env"
