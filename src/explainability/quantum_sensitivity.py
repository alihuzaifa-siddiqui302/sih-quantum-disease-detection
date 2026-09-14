"""
src/explainability/quantum_sensitivity.py
==========================================
Quantum Bottleneck Sensitivity Analysis for QTL (SIH26139).
Computes the gradient attribution matrix:
    S_{j, k} = |∂⟨Z_j⟩ / ∂z_k|
for all 4 measurement qubits j ∈ {0..3} and all 6 input bottleneck qubits k ∈ {0..5},
averaged across held-out Brain MRI test samples.

Produces results/figures/quantum_bottleneck_importance.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.qtl_architecture import QuantumTransferModel, N_QUBITS, N_CLASSES
from src.preprocessing.image_pipeline import CLASS_NAMES, get_dataloaders

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
FIGURES_DIR    = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_quantum_sensitivity_analysis(n_samples_per_class: int = 25):
    print("[Quantum Sensitivity] Initializing QTL model...")
    device = torch.device("cpu")
    model = QuantumTransferModel(n_classes=N_CLASSES, freeze_backbone=False, pretrained=False)
    ckpt = CHECKPOINT_DIR / "qtl_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    print("[Quantum Sensitivity] Loading test samples from Brain MRI dataset...")
    loaders = get_dataloaders(batch_size=32)
    test_loader = loaders["test"]

    # Collect samples balanced across classes
    samples_per_class = {c: [] for c in range(N_CLASSES)}
    total_target = n_samples_per_class * N_CLASSES

    for imgs, lbls in test_loader:
        for img, lbl in zip(imgs, lbls):
            cls_idx = int(lbl.item())
            if len(samples_per_class[cls_idx]) < n_samples_per_class:
                samples_per_class[cls_idx].append(img.unsqueeze(0))
        if sum(len(v) for v in samples_per_class.values()) >= total_target:
            break

    all_imgs = []
    for c in range(N_CLASSES):
        all_imgs.extend(samples_per_class[c])
    X_batch = torch.cat(all_imgs, dim=0).to(device)  # [N, 3, 224, 224]
    N = X_batch.shape[0]
    print(f"[Quantum Sensitivity] Evaluating sensitivity across N={N} test samples...")

    # Extract classical features from backbone
    with torch.no_grad():
        backbone_feats = model.backbone(X_batch).flatten(start_dim=1)  # [N, 512]
        # Bottleneck forward (linear + tanh + pi-scale)
        z_inputs = model.bottleneck(backbone_feats)                     # [N, 6]

    # Compute Jacobian |d<Z_j> / dz_k| via autograd
    jacobian_matrix = np.zeros((N_CLASSES, N_QUBITS))

    # Process sample-by-sample for exact per-instance Jacobians
    for i in range(N):
        z_sample = z_inputs[i:i+1].clone().detach().requires_grad_(True)
        # quantum layer forward
        q_expvals = model.quantum_layer(z_sample)  # [1, 4]

        for j in range(N_CLASSES):
            model.zero_grad()
            if z_sample.grad is not None:
                z_sample.grad.zero_()
            grad_j = torch.autograd.grad(
                q_expvals[0, j], z_sample, retain_graph=True, create_graph=False
            )[0]  # [1, 6]
            jacobian_matrix[j, :] += torch.abs(grad_j).squeeze().cpu().numpy()

    # Average across N samples
    jacobian_matrix /= N

    # Overall qubit importance (summed across all classes)
    total_qubit_importance = np.sum(jacobian_matrix, axis=0)
    relative_importance = (total_qubit_importance / np.sum(total_qubit_importance)) * 100.0

    print("\n[Quantum Sensitivity] Sensitivity Matrix S_{j,k} = |d<Z_j> / dz_k|:")
    for j, cname in enumerate(CLASS_NAMES):
        vals = "  ".join(f"q{k}: {jacobian_matrix[j, k]:.4f}" for k in range(N_QUBITS))
        print(f"  Class {cname:11s} -> {vals}")

    print("\n[Quantum Sensitivity] Relative Qubit Importance:")
    for k in range(N_QUBITS):
        print(f"  Qubit {k} (z_{k}): {relative_importance[k]:.2f}% (Total sensitivity: {total_qubit_importance[k]:.4f})")

    # ── Publication-quality visualization ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    plt.subplots_adjust(wspace=0.3)

    # Subplot 1: Sensitivity Heatmap (Outputs Z_j vs Input Qubits z_k)
    qubit_labels = [f"Qubit {k}\n($z_{k}$)" for k in range(N_QUBITS)]
    class_labels = [f"⟨Z_{j}⟩: {CLASS_NAMES[j].capitalize()}" for j in range(N_CLASSES)]

    sns.heatmap(
        jacobian_matrix,
        annot=True,
        fmt=".3f",
        cmap="mako",
        xticklabels=qubit_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Mean Gradient Magnitude $|\\partial \\langle Z_j \\rangle / \\partial z_k|$"},
        ax=ax1,
        linewidths=0.5,
        linecolor="white",
    )
    ax1.set_title("Quantum Transfer Circuit Sensitivity Matrix\n(Output Expectation Values vs. Input Bottleneck Features)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Input Bottleneck Qubits ($k \\in \\{0..5\\}$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Measured Qubits / Class Logits ($j \\in \\{0..3\\}$)", fontsize=10, fontweight="bold")

    # Subplot 2: Qubit Importance Bar Chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.85, N_QUBITS))
    bars = ax2.bar(range(N_QUBITS), total_qubit_importance, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax2.set_xticks(range(N_QUBITS))
    ax2.set_xticklabels([f"Qubit {k}\n($z_{k}$)" for k in range(N_QUBITS)], fontsize=10)
    ax2.set_ylabel("Total Gradient Magnitude $\\sum_j |\\partial \\langle Z_j \\rangle / \\partial z_k|$", fontsize=10, fontweight="bold")
    ax2.set_title("Bottleneck Qubit Expressivity & Attribution\nRelative Share of Quantum Output Gradient", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, pct, tot in zip(bars, relative_importance, total_qubit_importance):
        height = bar.get_height()
        ax2.annotate(
            f"{pct:.1f}%\n({tot:.3f})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle("SIH26139 Quantum Bottleneck Sensitivity Analysis\nValidating Non-Degenerate Information Flow Through 6-Qubit PQC", fontsize=13, fontweight="bold", y=0.98)

    out_file = FIGURES_DIR / "quantum_bottleneck_importance.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Quantum Sensitivity] Figure saved successfully to {out_file}")


if __name__ == "__main__":
    run_quantum_sensitivity_analysis()
