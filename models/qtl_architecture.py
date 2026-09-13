"""
models/qtl_architecture.py
===========================
Engine 2 — Quantum Transfer Learning (QTL) for Brain MRI 4-class classification.

Full Architecture
-----------------
  ResNet-18 (pretrained, backbone frozen in Phase 1)
    └─> Remove final fc layer  →  512-dim feature vector
        └─> Linear(512 → 6)          [bottleneck, π-scaled via Tanh+scale]
            └─> AngleEmbedding        [6-qubit Ry/Rz angle embedding]
                └─> StronglyEntanglingLayers(wires=6, n_layers=3)
                    └─> [qml.expval(PauliZ(i)) for i in range(4)]   [4 measurements]
                        └─> Linear(4 → 4) + output                  [4-class MRI]

Quantum Circuit Details
-----------------------
Device: pennylane.device("lightning.qubit", wires=6)
  - lightning.qubit: C++-accelerated statevector simulator (fastest CPU device)
  - diff_method="adjoint": O(n_params) gradient computation using the adjoint
    differentiation method — no finite differences, scales to ~1000 parameters.

AngleEmbedding:
  - Encodes 6 bottleneck features as rotation angles on 6 qubits
  - Uses Ry rotations by default (configurable)

StronglyEntanglingLayers:
  - 3 layers of parameterized single-qubit rotations + CNOT entanglement
  - Each layer: Rot(θ,φ,ω) on each qubit + ring CNOT entanglement
  - Total trainable parameters: 3 layers × 6 qubits × 3 angles = 54 parameters

Training Phases
---------------
Phase 1 (default): Backbone frozen, train only:
    - bottleneck Linear(512→6)
    - StronglyEntanglingLayers weights (54 params)
    - output Linear(4→4)

Phase 2 (fine-tuning): Unfreeze last ResNet block + train everything.

Usage
-----
    from models.qtl_architecture import QuantumTransferModel
    model = QuantumTransferModel(n_classes=4, freeze_backbone=True)
    output = model(torch.randn(4, 3, 224, 224))  # [4, 4]

    # Or from CLI:
    python -m models.qtl_architecture
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision.models as tv_models

# ─── Constants ────────────────────────────────────────────────────────────────
N_QUBITS    = 6      # bottleneck dimension = number of qubits
N_QNN_LAYERS = 3     # StronglyEntanglingLayers depth
N_CLASSES   = 4      # glioma / meningioma / notumor / pituitary
BACKBONE_DIM = 512   # ResNet-18 penultimate feature dimension

ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PennyLane Quantum Circuit
# ═══════════════════════════════════════════════════════════════════════════

# Instantiate the lightning.qubit device once at module level for efficiency.
# lightning.qubit uses a C++ statevector backend — fastest CPU-side PennyLane device.
_dev = qml.device("lightning.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch", diff_method="adjoint")
def _quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """
    6-qubit Parameterized Quantum Circuit (PQC).

    Step 1 — AngleEmbedding:
        Encodes the 6 bottleneck features as Ry rotation angles.
        |ψ_data⟩ = Ry(x_0)|0⟩ ⊗ Ry(x_1)|0⟩ ⊗ ... ⊗ Ry(x_5)|0⟩

    Step 2 — StronglyEntanglingLayers:
        3 layers of parameterized entangled rotations.
        Each layer applies:
          (a) Rot(θ, φ, ω) on each qubit — arbitrary SU(2) rotation
          (b) CNOT ring entanglement between neighboring qubits
        This creates rich entangled states for expressive quantum learning.

    Step 3 — Measurement:
        Measure Pauli-Z expectation value on first 4 qubits.
        Returns values in [-1, 1] — used as 4 class logits.

    diff_method="adjoint":
        Adjoint differentiation computes exact parameter-shift-free gradients
        in O(n_params) forward+backward passes. Avoids exponential cost of
        finite-differences and the 2×overhead of parameter-shift rule.

    Args:
        inputs:  Tensor [N_QUBITS]  — bottleneck features (angle-encoded)
        weights: Tensor [N_QNN_LAYERS, N_QUBITS, 3] — StronglyEntanglingLayers weights

    Returns:
        List of N_CLASSES PauliZ expectation values, each in [-1.0, +1.0].
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_CLASSES)]


def _build_torch_layer() -> qml.qnn.TorchLayer:
    """
    Wrap the quantum circuit as a nn.Module-compatible TorchLayer.
    weight_shapes specifies the shape of trainable quantum parameters.
    """
    weight_shapes = {
        "weights": (N_QNN_LAYERS, N_QUBITS, 3),  # 3×6×3 = 54 trainable params
    }
    return qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Bottleneck Module
# ═══════════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """
    Linear bottleneck: maps ResNet-18 features (512-dim) to N_QUBITS (6) values.

    The output is passed through Tanh and scaled to [-π, π], then shifted
    to [0, π] to serve as valid rotation angles for AngleEmbedding.

    Transformation: y = (Tanh(Wx + b) + 1) * π/2  →  output ∈ [0, π]
    """

    def __init__(self, in_features: int = BACKBONE_DIM, out_features: int = N_QUBITS):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.scale = math.pi / 2.0  # maps [-1,1] → [-π/2, π/2]; after +1 → [0, π]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512]  →  out: [B, 6]  ∈ [0, π]
        return (torch.tanh(self.fc(x)) + 1.0) * self.scale


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Full Hybrid Model
# ═══════════════════════════════════════════════════════════════════════════

class QuantumTransferModel(nn.Module):
    """
    Hybrid Quantum-Classical Transfer Learning model for Brain MRI classification.

    Architecture:
        ResNet-18 backbone (pretrained ImageNet weights, fc removed)
        → Bottleneck Linear(512 → 6) with Tanh + π-scaling
        → PennyLane TorchLayer (6 qubits, AngleEmbedding + StronglyEntanglingLayers)
        → Linear(4 → 4) output layer
        → LogSoftmax (for NLLLoss) or raw logits (for CrossEntropyLoss)

    Args:
        n_classes:       Number of output classes (4 for MRI).
        freeze_backbone: If True (Phase 1), all ResNet layers are frozen.
                         Set False for Phase 2 fine-tuning.
        pretrained:      Load ImageNet pretrained weights for ResNet-18.
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Classical backbone: ResNet-18 ──────────────────────────────────
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        # Remove the final fully-connected layer; keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 512, 1, 1]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[QTL] ResNet-18 backbone FROZEN ({sum(p.numel() for p in self.backbone.parameters())} params)")

        # ── Bottleneck: 512 → N_QUBITS ────────────────────────────────────
        self.bottleneck = Bottleneck(in_features=BACKBONE_DIM, out_features=N_QUBITS)

        # ── Quantum layer ─────────────────────────────────────────────────
        self.quantum_layer = _build_torch_layer()

        # ── Output classifier ─────────────────────────────────────────────
        self.output_layer = nn.Linear(n_classes, n_classes)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[QTL] Trainable parameters: {n_trainable}")
        print(f"[QTL] Quantum circuit: {N_QUBITS} qubits, {N_QNN_LAYERS} layers, 54 quantum params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image batch [B, 3, 224, 224]

        Returns:
            Class logits [B, 4]
        """
        # Step 1: ResNet-18 backbone feature extraction
        feats = self.backbone(x)          # [B, 512, 1, 1]
        feats = feats.flatten(start_dim=1)  # [B, 512]

        # Step 2: Bottleneck → 6 angle-encoded features ∈ [0, π]
        angles = self.bottleneck(feats)   # [B, 6]

        # Step 3: Quantum forward pass (processed sample-by-sample by TorchLayer)
        q_out = self.quantum_layer(angles)  # [B, 4]  values ∈ [-1, 1]

        # Step 4: Classical output layer → class logits
        logits = self.output_layer(q_out)   # [B, 4]

        return logits

    def unfreeze_backbone(self, unfreeze_layers: int = 1) -> None:
        """
        Phase 2 fine-tuning: unfreeze the last `unfreeze_layers` ResNet layer groups
        that actually contain learnable parameters.

        ResNet-18 backbone children (after fc removal):
          [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        avgpool has NO parameters — we skip parameter-less modules and unfreeze
        the correct number of layers counting from the end of parametrized ones.

        Args:
            unfreeze_layers: Number of parametrized layer groups from the end to unfreeze.
        """
        # Filter to only children that have learnable parameters
        children_with_params = [
            child for child in self.backbone.children()
            if sum(1 for _ in child.parameters()) > 0
        ]
        for child in children_with_params[-unfreeze_layers:]:
            for param in child.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[QTL] Unfrozen last {unfreeze_layers} parametrized backbone layer(s). Total trainable: {n_trainable}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Circuit Diagram Export
# ═══════════════════════════════════════════════════════════════════════════

def save_circuit_diagram(out_dir: Optional[Path] = None) -> str:
    """
    Draw and save the PennyLane PQC circuit diagram.
    Returns the diagram string.
    """
    if out_dir is None:
        out_dir = ROOT.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs and weights for drawing
    dummy_inputs  = np.zeros(N_QUBITS)
    dummy_weights = np.zeros((N_QNN_LAYERS, N_QUBITS, 3))

    drawer = qml.draw(_quantum_circuit)
    diagram_str = drawer(dummy_inputs, dummy_weights)

    out_path = out_dir / "qtl_circuit.txt"
    out_path.write_text(
        f"Quantum Transfer Learning — PQC Diagram\n"
        f"Device: lightning.qubit | Wires: {N_QUBITS} | diff_method: adjoint\n"
        f"AngleEmbedding (Ry) + StronglyEntanglingLayers ({N_QNN_LAYERS} layers)\n"
        f"{'='*70}\n"
        f"{diagram_str}\n",
        encoding="utf-8",
    )
    print(f"[QTL] PQC diagram saved to {out_path}")
    return diagram_str


# ═══════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("QuantumTransferModel — Architecture Smoke Test")
    print("=" * 60)

    model = QuantumTransferModel(n_classes=4, freeze_backbone=True, pretrained=False)
    model.eval()

    batch_size = 2  # small for CPU speed
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\nRunning forward pass: input={list(dummy_input.shape)} ...")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {list(output.shape)}")
    assert output.shape == (batch_size, 4), \
        f"[FAIL] Expected ({batch_size}, 4), got {output.shape}"
    print("[OK] Forward pass assertion PASSED: output shape == (batch_size, 4)")

    # Save circuit diagram
    diagram = save_circuit_diagram()
    print(f"\nCircuit:\n{diagram}")
