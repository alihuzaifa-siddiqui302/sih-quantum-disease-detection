"""
src/classical/baseline_imaging.py
===================================
Pure classical ResNet-18 baseline for Brain MRI 4-class classification (SIH26139).
Architecture: ResNet-18 backbone + Linear(512 -> 4) directly (no quantum head).

Training mirrors the QTL two-stage setup for fair comparison:
  Stage 1: Frozen backbone, train final FC 10 epochs, Adam lr=1e-3
  Stage 2: Unfreeze last 2 ResNet layers, train 20 epochs (early stop patience=5)
           Differential LRs: backbone lr=1e-5, head lr=1e-3

Checkpoint: models/checkpoints/classical_resnet_best.pth
History:    results/metrics/classical_resnet_history.json
Predictions: results/metrics/classical_resnet_test_preds.npy

Usage
-----
    python -m src.classical.baseline_imaging
    python -m src.classical.baseline_imaging --epochs1 5 --epochs2 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METRICS_DIR    = ROOT / "results" / "metrics"
FIGURES_DIR    = ROOT / "results" / "figures"

for _d in (CHECKPOINT_DIR, METRICS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

N_CLASSES = 4


# ─── Model ────────────────────────────────────────────────────────────────────

class ClassicalResNet18(nn.Module):
    """
    ResNet-18 with a Linear(512 -> 4) classifier head.
    Identical backbone setup to QuantumTransferModel, but with a classical head
    instead of the bottleneck + quantum layer.
    """

    def __init__(self, n_classes: int = N_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tv_models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])   # [B, 512, 1, 1]
        self.classifier = nn.Linear(512, n_classes)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[ClassicalResNet] Total parameters: {n_total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x).flatten(start_dim=1)  # [B, 512]
        return self.classifier(feats)                    # [B, 4]

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ClassicalResNet] Backbone FROZEN. Trainable: {n:,}")

    def unfreeze_backbone(self, n_layers: int = 2) -> None:
        children_with_params = [
            c for c in self.backbone.children()
            if sum(1 for _ in c.parameters()) > 0
        ]
        for child in children_with_params[-n_layers:]:
            for p in child.parameters():
                p.requires_grad = True
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ClassicalResNet] Unfrozen last {n_layers} layers. Trainable: {n:,}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _f2(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, beta=2, average="macro", zero_division=0))


def _eval_model(model, loader, criterion, device="cpu") -> tuple[float, float, float]:
    model.eval()
    total_loss, preds_all, true_all = 0.0, [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, lbls).item()
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            true_all.append(lbls.cpu().numpy())
    yp = np.concatenate(preds_all)
    yt = np.concatenate(true_all)
    avg_loss = total_loss / max(len(list(loader)), 1)
    return avg_loss, float(accuracy_score(yt, yp)), _f2(yt, yp)


# ─── Stage 1 ─────────────────────────────────────────────────────────────────

def train_stage1(
    model: ClassicalResNet18,
    train_loader, val_loader,
    n_epochs: int = 10,
    lr: float = 1e-3,
) -> dict:
    model.freeze_backbone()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_recall = -1.0
    best_ckpt = CHECKPOINT_DIR / "classical_resnet_stage1_best.pth"
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f2": []}

    print(f"\n  Stage 1: {n_epochs} epochs, lr={lr}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        model.backbone.eval()   # backbone stays frozen / eval mode
        t0 = time.perf_counter()
        epoch_loss, n_b = 0.0, 0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_loss, val_acc, val_f2 = _eval_model(model, val_loader, criterion)
        history["train_loss"].append(round(epoch_loss / max(n_b, 1), 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f2"].append(round(val_f2, 4))

        mark = ""
        if val_f2 > best_recall:
            best_recall = val_f2
            torch.save(model.state_dict(), best_ckpt)
            mark = " [CKPT]"
        print(f"  Epoch {epoch:02d}/{n_epochs}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f2={val_f2:.4f}  {elapsed:.0f}s{mark}")

    return history


# ─── Stage 2 ─────────────────────────────────────────────────────────────────

def train_stage2(
    model: ClassicalResNet18,
    train_loader, val_loader,
    n_epochs: int = 20,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-3,
    patience: int = 5,
) -> dict:
    model.unfreeze_backbone(n_layers=2)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.classifier.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_f2 = -1.0
    patience_ctr = 0
    best_ckpt = CHECKPOINT_DIR / "classical_resnet_best.pth"
    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f2": [],
        "stopped_epoch": n_epochs,
    }

    print(f"\n  Stage 2: up to {n_epochs} epochs  lr_backbone={lr_backbone} lr_head={lr_head}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_loss, n_b = 0.0, 0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_loss, val_acc, val_f2 = _eval_model(model, val_loader, criterion)
        history["train_loss"].append(round(epoch_loss / max(n_b, 1), 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f2"].append(round(val_f2, 4))

        mark = ""
        if val_f2 > best_f2:
            best_f2 = val_f2
            patience_ctr = 0
            torch.save(model.state_dict(), best_ckpt)
            mark = " [CKPT]"
        else:
            patience_ctr += 1

        print(f"  Epoch {epoch:02d}/{n_epochs}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f2={val_f2:.4f}  "
              f"{elapsed:.0f}s{mark}  (patience {patience_ctr}/{patience})")

        if patience_ctr >= patience:
            history["stopped_epoch"] = epoch
            print(f"  Early stopping at epoch {epoch}.")
            break

    return history


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SIH26139 Classical ResNet-18 Baseline")
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=20)
    parser.add_argument("--batch",   type=int, default=32)
    args = parser.parse_args()

    from src.preprocessing.image_pipeline import get_dataloaders
    print("\n[ClassicalResNet] Loading Brain MRI DataLoaders...")
    loaders = get_dataloaders(batch_size=args.batch)

    model = ClassicalResNet18(n_classes=4, pretrained=True)
    total_start = time.perf_counter()

    # Stage 1
    s1_history = train_stage1(model, loaders["train"], loaders["val"], n_epochs=args.epochs1)

    # Load best S1 weights
    s1_ckpt = CHECKPOINT_DIR / "classical_resnet_stage1_best.pth"
    if s1_ckpt.exists():
        model.load_state_dict(torch.load(s1_ckpt, map_location="cpu"))

    # Stage 2
    s2_history = train_stage2(model, loaders["train"], loaders["val"], n_epochs=args.epochs2)

    total_time = time.perf_counter() - total_start
    print(f"\n[ClassicalResNet] Total wall-clock: {total_time/60:.1f} min")

    # ── Test evaluation ────────────────────────────────────────────────────
    best_ckpt = CHECKPOINT_DIR / "classical_resnet_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    model.eval()

    test_preds, test_true = [], []
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for imgs, lbls in loaders["test"]:
            logits = model(imgs)
            test_preds.append(logits.argmax(dim=1).cpu().numpy())
            test_true.append(lbls.cpu().numpy())
    inf_time = time.perf_counter() - t_inf_start
    tp = np.concatenate(test_preds)
    tt = np.concatenate(test_true)
    test_acc = float(accuracy_score(tt, tp))
    test_f2  = _f2(tt, tp)
    latency  = inf_time / len(tt) * 1000

    print(f"[ClassicalResNet] Test accuracy={test_acc:.4f}  F2={test_f2:.4f}  latency={latency:.2f}ms/sample")

    np.save(METRICS_DIR / "classical_resnet_test_preds.npy", tp)
    np.save(METRICS_DIR / "classical_resnet_test_true.npy",  tt)

    # Save combined history
    combined_history = {
        "stage1": s1_history,
        "stage2": s2_history,
        "val_f2": s1_history["val_f2"] + s2_history["val_f2"],
        "val_loss": s1_history["val_loss"] + s2_history["val_loss"],
        "train_loss": s1_history["train_loss"] + s2_history["train_loss"],
    }
    hist_path = METRICS_DIR / "classical_resnet_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    result = {
        "model": "classical_resnet18",
        "dataset": "brain_mri",
        "total_train_time_s": round(total_time, 1),
        "test": {
            "accuracy": test_acc,
            "f2": test_f2,
            "latency_ms_per_sample": round(latency, 3),
            "report": classification_report(tt, tp, output_dict=True, zero_division=0),
        },
    }
    result_path = METRICS_DIR / "classical_resnet.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[ClassicalResNet] Results saved to {result_path}")


if __name__ == "__main__":
    main()
