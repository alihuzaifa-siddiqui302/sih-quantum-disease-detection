"""
src/quantum/train_qtl.py
=========================
Two-stage Quantum Transfer Learning training for Brain MRI (4-class).

STAGE 1 — Frozen Backbone Warmup (10 epochs)
  ResNet-18 backbone is frozen. Only train:
    bottleneck Linear(512->6) + quantum TorchLayer (54 params) + output Linear(4->4)
  Checkpoint: best val macro-Recall saved to models/checkpoints/qtl_stage1_best.pth

STAGE 2 — Joint Fine-Tuning (up to 20 epochs, early stop on val F2)
  Unfreeze last 2 ResNet layer groups (layer3 + layer4).
  Differential learning rates:
    backbone params:                 lr=1e-5
    quantum / bottleneck / output:   lr=1e-3
  Early stopping: patience=5 on val F2-score (beta=2, macro).
  Checkpoint: best weights saved to models/checkpoints/qtl_best.pth
  Training curves: results/figures/training_curves_qtl.png

FEATURE CACHING (CPU feasibility)
  ResNet-18 backbone extracts 512-dim features.
  Cached once per split to data/processed/mri_features_{split}.npz.
  Subsequent epochs skip backbone inference entirely, training only the
  quantum head on cached vectors. This reduces per-epoch time ~10x on CPU.

Usage
-----
    python -m src.quantum.train_qtl              # full Stage 1 + Stage 2
    python -m src.quantum.train_qtl --stage 1    # Stage 1 only
    python -m src.quantum.train_qtl --epochs1 5 --epochs2 10  # shorter run
    python -m src.quantum.train_qtl --no-cache   # disable feature caching
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import fbeta_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
FIGURES_DIR   = ROOT / "results" / "figures"
METRICS_DIR   = ROOT / "results" / "metrics"

for _d in (CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from models.qtl_architecture import QuantumTransferModel, N_QUBITS


# ═══════════════════════════════════════════════════════════════════════════
# Feature caching helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_and_cache_features(
    model: QuantumTransferModel,
    loader: DataLoader,
    split: str,
    force: bool = False,
) -> TensorDataset:
    """
    Pass all images through the FROZEN ResNet-18 backbone once and cache
    the 512-dim feature vectors to data/processed/mri_features_{split}.npz.

    On subsequent calls, loads from cache (skips backbone inference entirely).

    Args:
        model:  QuantumTransferModel (backbone must be frozen).
        loader: DataLoader for the split.
        split:  One of 'train', 'val', 'test'.
        force:  If True, re-extract even if cache exists.

    Returns:
        TensorDataset of (features [N, 512], labels [N]).
    """
    cache_path = PROCESSED_DIR / f"mri_features_{split}.npz"

    if cache_path.exists() and not force:
        print(f"  [cache] Loading cached features: {cache_path}")
        d = np.load(cache_path)
        feats = torch.from_numpy(d["features"])
        labels = torch.from_numpy(d["labels"]).long()
        return TensorDataset(feats, labels)

    print(f"  [cache] Extracting ResNet-18 features for '{split}' split...")
    model.backbone.eval()
    all_feats, all_labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, lbls in loader:
            feats = model.backbone(imgs)          # [B, 512, 1, 1]
            feats = feats.flatten(start_dim=1)    # [B, 512]
            all_feats.append(feats.cpu())
            all_labels.append(lbls.cpu())
    elapsed = time.perf_counter() - t0

    feats_t  = torch.cat(all_feats,  dim=0)
    labels_t = torch.cat(all_labels, dim=0)

    np.savez(cache_path, features=feats_t.numpy(), labels=labels_t.numpy())
    print(f"  [cache] Saved {feats_t.shape[0]} vectors to {cache_path} ({elapsed:.1f}s)")
    return TensorDataset(feats_t, labels_t)


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F2-score (beta=2, weights recall twice as much as precision)."""
    return float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def _evaluate_head(
    model: QuantumTransferModel,
    feat_loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the quantum head (bottleneck + quantum_layer + output_layer) on
    cached features. Returns (loss, macro_recall, f2, y_true, y_pred).
    """
    model.bottleneck.eval()
    model.quantum_layer.eval()
    model.output_layer.eval()
    total_loss, all_preds, all_true = 0.0, [], []
    n_batches = 0
    with torch.no_grad():
        for feats, labels in feat_loader:
            angles  = model.bottleneck(feats)
            q_out   = model.quantum_layer(angles)
            logits  = model.output_layer(q_out)
            loss    = criterion(logits, labels)
            total_loss += loss.item()
            n_batches  += 1
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, _macro_recall(y_true, y_pred), _f2_score(y_true, y_pred), y_true, y_pred


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Frozen backbone warmup
# ═══════════════════════════════════════════════════════════════════════════

def train_stage1(
    model: QuantumTransferModel,
    train_feat_ds: TensorDataset,
    val_feat_ds: TensorDataset,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict:
    """
    Train bottleneck + quantum_layer + output on cached ResNet-18 features.
    Checkpoints best model (val macro-Recall) to models/checkpoints/qtl_stage1_best.pth.

    Returns history dict with per-epoch train_loss, val_loss, val_recall, val_f2.
    """
    train_loader = DataLoader(train_feat_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_feat_ds,   batch_size=batch_size, shuffle=False)

    # Only quantum head parameters are trained
    head_params = (
        list(model.bottleneck.parameters()) +
        list(model.quantum_layer.parameters()) +
        list(model.output_layer.parameters())
    )
    optimizer = optim.Adam(head_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_recall = -1.0
    best_ckpt   = CHECKPOINT_DIR / "qtl_stage1_best.pth"
    history = {"train_loss": [], "val_loss": [], "val_recall": [], "val_f2": []}

    print(f"\n  Stage 1: {n_epochs} epochs, lr={lr}, batch={batch_size}")
    for epoch in range(1, n_epochs + 1):
        model.bottleneck.train()
        model.quantum_layer.train()
        model.output_layer.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for feats, labels in train_loader:
            optimizer.zero_grad()
            angles = model.bottleneck(feats)
            q_out  = model.quantum_layer(angles)
            logits = model.output_layer(q_out)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        elapsed = time.perf_counter() - t0
        avg_train_loss = epoch_loss / max(n_batches, 1)

        val_loss, val_recall, val_f2, _, _ = _evaluate_head(model, val_loader, criterion)
        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_recall"].append(round(val_recall, 4))
        history["val_f2"].append(round(val_f2, 4))

        ckpt_mark = ""
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), best_ckpt)
            ckpt_mark = " [CKPT]"

        print(
            f"  Epoch {epoch:02d}/{n_epochs}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_recall={val_recall:.4f}  val_f2={val_f2:.4f}  "
            f"{elapsed:.0f}s{ckpt_mark}"
        )

    print(f"  Stage 1 complete. Best val_recall={best_recall:.4f} -> {best_ckpt}")
    return history


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Joint fine-tuning with differential LRs
# ═══════════════════════════════════════════════════════════════════════════

def train_stage2(
    model: QuantumTransferModel,
    train_loader_full: DataLoader,   # original image DataLoader (backbone re-active)
    val_loader_full: DataLoader,
    n_epochs: int = 20,
    lr_backbone: float = 1e-5,
    lr_quantum: float = 1e-3,
    patience: int = 5,
) -> dict:
    """
    Joint fine-tuning with differential learning rates.

    Backbone layers (layer3 + layer4 unfrozen) use lr_backbone=1e-5.
    Bottleneck, quantum_layer, output_layer use lr_quantum=1e-3.

    Early stopping: patience=5 epochs on val F2-score (no improvement).
    Best model saved to models/checkpoints/qtl_best.pth.

    NOTE: This stage uses FULL IMAGE DataLoaders (not cached features)
    because the backbone is now unfrozen and its weights change each epoch.

    Returns history dict.
    """
    model.unfreeze_backbone(unfreeze_layers=2)

    # Separate param groups for differential LRs
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.bottleneck.parameters()) +
        list(model.quantum_layer.parameters()) +
        list(model.output_layer.parameters())
    )

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_quantum},
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f2     = -1.0
    patience_ctr = 0
    best_ckpt   = CHECKPOINT_DIR / "qtl_best.pth"
    history = {
        "train_loss": [], "val_loss": [], "val_recall": [], "val_f2": [],
        "stopped_epoch": n_epochs,
    }

    print(f"\n  Stage 2: up to {n_epochs} epochs, lr_backbone={lr_backbone}, lr_quantum={lr_quantum}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.perf_counter()

        for imgs, labels in train_loader_full:
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        elapsed = time.perf_counter() - t0
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on val
        model.eval()
        val_loss_sum, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader_full:
                logits = model(imgs)
                val_loss_sum += criterion(logits, labels).item()
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_true.append(labels.cpu().numpy())
        vl = val_loss_sum / max(len(list(val_loader_full)), 1)
        vp = np.concatenate(val_preds)
        vt = np.concatenate(val_true)
        val_recall = _macro_recall(vt, vp)
        val_f2     = _f2_score(vt, vp)

        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(vl, 5))
        history["val_recall"].append(round(val_recall, 4))
        history["val_f2"].append(round(val_f2, 4))

        ckpt_mark = ""
        if val_f2 > best_f2:
            best_f2 = val_f2
            patience_ctr = 0
            torch.save(model.state_dict(), best_ckpt)
            ckpt_mark = " [CKPT]"
        else:
            patience_ctr += 1

        print(
            f"  Epoch {epoch:02d}/{n_epochs}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={vl:.4f}  "
            f"val_recall={val_recall:.4f}  val_f2={val_f2:.4f}  "
            f"{elapsed:.0f}s{ckpt_mark}  (patience {patience_ctr}/{patience})"
        )

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch} (val F2 no improvement for {patience} epochs)")
            history["stopped_epoch"] = epoch
            break

    print(f"  Stage 2 complete. Best val_f2={best_f2:.4f} -> {best_ckpt}")
    return history


# ═══════════════════════════════════════════════════════════════════════════
# Training curve plotter
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    s1_history: dict,
    s2_history: dict | None,
    classical_history: dict | None = None,
    out_path: Path | None = None,
) -> None:
    """Save loss and F2-score training curves for QTL (both stages) and optional classical."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  [plot] matplotlib not available — skipping curve plot")
        return

    if out_path is None:
        out_path = FIGURES_DIR / "training_curves_qtl.png"

    s1_epochs = list(range(1, len(s1_history["val_f2"]) + 1))
    s2_start = len(s1_epochs) + 1
    s2_epochs = (
        list(range(s2_start, s2_start + len(s2_history["val_f2"])))
        if s2_history else []
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("QTL Training Curves — SIH26139", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(s1_epochs, s1_history["train_loss"], "b-",  label="QTL train loss (S1)")
    ax.plot(s1_epochs, s1_history["val_loss"],   "b--", label="QTL val loss (S1)")
    if s2_history:
        ax.plot(s2_epochs, s2_history["train_loss"], "g-",  label="QTL train loss (S2)")
        ax.plot(s2_epochs, s2_history["val_loss"],   "g--", label="QTL val loss (S2)")
    if s1_epochs and s2_epochs:
        ax.axvline(s1_epochs[-1] + 0.5, color="gray", linestyle=":", alpha=0.6, label="Stage boundary")
    if classical_history:
        cl_epochs = list(range(1, len(classical_history.get("val_loss", [])) + 1))
        ax.plot(cl_epochs, classical_history["val_loss"], "r--", label="Classical val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── F2-score ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(s1_epochs, s1_history["val_f2"], "b-o", markersize=4, label="QTL val F2 (S1)")
    if s2_history:
        ax.plot(s2_epochs, s2_history["val_f2"], "g-o", markersize=4, label="QTL val F2 (S2)")
    if classical_history:
        cl_epochs = list(range(1, len(classical_history.get("val_f2", [])) + 1))
        ax.plot(cl_epochs, classical_history.get("val_f2", []), "r-s", markersize=4, label="Classical val F2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F2-Score")
    ax.set_title("Validation F2-Score (β=2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Training curves saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 QTL Training")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run only stage 1 or 2 (default: both)")
    parser.add_argument("--epochs1", type=int, default=10, help="Stage 1 epochs")
    parser.add_argument("--epochs2", type=int, default=20, help="Stage 2 epochs")
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--seed",    type=int, default=42, help="Random seed for deterministic reproducibility")
    parser.add_argument("--no-cache", action="store_true", help="Re-extract features even if cached")
    args = parser.parse_args()

    # Seed all PRNGs for deterministic reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from src.preprocessing.image_pipeline import get_dataloaders
    print(f"\n[QTL Train] Loading Brain MRI DataLoaders (seed={args.seed})...")
    loaders = get_dataloaders(batch_size=args.batch, seed=args.seed)

    print("[QTL Train] Building QuantumTransferModel (backbone FROZEN)...")
    model = QuantumTransferModel(n_classes=4, freeze_backbone=True, pretrained=True)
    model.eval()

    s1_history, s2_history = None, None
    total_start = time.perf_counter()

    # ── Stage 1 ──────────────────────────────────────────────────────────
    if args.stage in (None, 1):
        print("\n[Stage 1] Extracting/loading cached ResNet-18 features...")
        force = args.no_cache
        train_feat_ds = _extract_and_cache_features(model, loaders["train"], "train", force=force)
        val_feat_ds   = _extract_and_cache_features(model, loaders["val"],   "val",   force=force)

        print(f"[Stage 1] Training quantum head on {len(train_feat_ds)} samples...")
        s1_history = train_stage1(
            model, train_feat_ds, val_feat_ds,
            n_epochs=args.epochs1, batch_size=args.batch,
        )

        s1_results_path = METRICS_DIR / "qtl_stage1_history.json"
        with open(s1_results_path, "w", encoding="utf-8") as f:
            json.dump(s1_history, f, indent=2)
        print(f"[Stage 1] History saved to {s1_results_path}")

    # ── Stage 2 ──────────────────────────────────────────────────────────
    if args.stage in (None, 2):
        # Load best Stage 1 weights before Stage 2
        s1_ckpt = CHECKPOINT_DIR / "qtl_stage1_best.pth"
        if s1_ckpt.exists():
            model.load_state_dict(torch.load(s1_ckpt, map_location="cpu"))
            print(f"\n[Stage 2] Loaded Stage 1 weights from {s1_ckpt}")
        else:
            print("[Stage 2] No Stage 1 checkpoint found — training from current weights")

        print("[Stage 2] Joint fine-tuning with differential LRs (full image pipeline)...")
        s2_history = train_stage2(
            model,
            loaders["train"], loaders["val"],
            n_epochs=args.epochs2,
        )

        s2_results_path = METRICS_DIR / "qtl_stage2_history.json"
        with open(s2_results_path, "w", encoding="utf-8") as f:
            json.dump(s2_history, f, indent=2)
        print(f"[Stage 2] History saved to {s2_results_path}")

    total_time = time.perf_counter() - total_start
    print(f"\n[QTL Train] Total wall-clock: {total_time/60:.1f} min")

    # ── Evaluate on test set ──────────────────────────────────────────────
    best_ckpt = CHECKPOINT_DIR / "qtl_best.pth"
    if not best_ckpt.exists():
        best_ckpt = CHECKPOINT_DIR / "qtl_stage1_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
        model.eval()
        test_preds, test_true = [], []
        with torch.no_grad():
            for imgs, lbls in loaders["test"]:
                logits = model(imgs)
                test_preds.append(logits.argmax(dim=1).cpu().numpy())
                test_true.append(lbls.cpu().numpy())
        tp = np.concatenate(test_preds)
        tt = np.concatenate(test_true)
        test_f2 = _f2_score(tt, tp)
        print(f"[QTL Train] Test F2-score: {test_f2:.4f}")

        # Save predictions for McNemar's test
        np.save(METRICS_DIR / "qtl_test_preds.npy", tp)
        np.save(METRICS_DIR / "qtl_test_true.npy", tt)

        # Save efficiency info
        eff = {
            "model": "qtl",
            "total_train_time_s": round(total_time, 1),
            "n_classes": 4,
            "test_f2": round(test_f2, 4),
        }
        with open(METRICS_DIR / "qtl_efficiency.json", "w") as f:
            json.dump(eff, f, indent=2)

    # ── Plot training curves ──────────────────────────────────────────────
    # Load classical history if available for overlay
    classical_history = None
    cl_hist_path = METRICS_DIR / "classical_resnet_history.json"
    if cl_hist_path.exists():
        with open(cl_hist_path, encoding="utf-8") as f:
            classical_history = json.load(f)

    plot_training_curves(
        s1_history or {"val_f2": [], "train_loss": [], "val_loss": []},
        s2_history,
        classical_history,
    )

    print("\n[QTL Train] DONE.")


if __name__ == "__main__":
    main()
