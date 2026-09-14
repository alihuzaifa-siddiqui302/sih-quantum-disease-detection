"""
src/explainability/gradcam.py
==============================
Grad-CAM (Gradient-weighted Class Activation Mapping) for Brain MRI models.
Visualizes spatial focus of the final conv layer (layer4) for both:
  1. Classical ResNet-18 baseline
  2. Hybrid Quantum Transfer Learning (QTL) model

Produces results/figures/gradcam_mri_explanations.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.qtl_architecture import QuantumTransferModel
from src.classical.baseline_imaging import ClassicalResNet18
from src.preprocessing.image_pipeline import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
MRI_TEST_DIR = ROOT / "data" / "raw" / "brain_mri" / "Testing"


class GradCAM:
    """Computes Grad-CAM heatmap for a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        # Global average pool of gradients across spatial dims (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        pred_prob = float(probs[0, target_class].item())
        return cam_resized, target_class, pred_prob


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays Grad-CAM heatmap onto RGB image [0, 255]."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * alpha + np.float32(img_rgb) * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def run_gradcam_pipeline():
    print("[Grad-CAM] Initializing models and weights...")
    device = torch.device("cpu")

    # Load Classical ResNet-18
    cl_model = ClassicalResNet18(n_classes=4, pretrained=False)
    cl_ckpt = CHECKPOINT_DIR / "classical_resnet_best.pth"
    cl_model.load_state_dict(torch.load(cl_ckpt, map_location=device))
    cl_model.eval()

    # Load QTL Model
    qtl_model = QuantumTransferModel(n_classes=4, freeze_backbone=False, pretrained=False)
    qtl_ckpt = CHECKPOINT_DIR / "qtl_best.pth"
    qtl_model.load_state_dict(torch.load(qtl_ckpt, map_location=device))
    qtl_model.eval()

    # Target final residual block in layer4 (index 7 of backbone Sequential)
    # backbone: 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, 5=layer2, 6=layer3, 7=layer4, 8=avgpool
    cl_target_layer = cl_model.backbone[7][-1]
    qtl_target_layer = qtl_model.backbone[7][-1]

    cl_cam_generator = GradCAM(cl_model, cl_target_layer)
    qtl_cam_generator = GradCAM(qtl_model, qtl_target_layer)

    transform_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    for row_idx, cname in enumerate(CLASS_NAMES):
        class_dir = MRI_TEST_DIR / cname
        img_paths = sorted(list(class_dir.glob("*.jpg")))
        if not img_paths:
            continue
        # Pick a representative sample (middle of dataset)
        sample_path = img_paths[len(img_paths) // 2]

        pil_img = Image.open(sample_path).convert("RGB")
        pil_resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_np = np.array(pil_resized)

        input_tensor = transform_tensor(pil_img).unsqueeze(0).to(device)

        # Classical Grad-CAM
        cl_cam, cl_pred, cl_prob = cl_cam_generator.generate_cam(input_tensor, target_class=row_idx)
        cl_overlay = overlay_cam_on_image(img_np, cl_cam)

        # QTL Grad-CAM
        qtl_cam, qtl_pred, qtl_prob = qtl_cam_generator.generate_cam(input_tensor, target_class=row_idx)
        qtl_overlay = overlay_cam_on_image(img_np, qtl_cam)

        # Plot Col 1: Original
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"True: {cname.upper()}\n{sample_path.name}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Plot Col 2: Classical ResNet-18
        axes[row_idx, 1].imshow(cl_overlay)
        axes[row_idx, 1].set_title(f"Classical ResNet-18\nPred: {CLASS_NAMES[cl_pred]} ({cl_prob*100:.1f}%)", fontsize=11)
        axes[row_idx, 1].axis("off")

        # Plot Col 3: QTL
        axes[row_idx, 2].imshow(qtl_overlay)
        axes[row_idx, 2].set_title(f"Hybrid QTL (Ours)\nPred: {CLASS_NAMES[qtl_pred]} ({qtl_prob*100:.1f}%)", fontsize=11, color="navy")
        axes[row_idx, 2].axis("off")

    cl_cam_generator.remove_hooks()
    qtl_cam_generator.remove_hooks()

    plt.suptitle("Brain MRI Clinical Explainability: Grad-CAM Saliency Maps\nClassical ResNet-18 vs. Hybrid Quantum Transfer Learning (QTL)", fontsize=14, fontweight="bold", y=0.98)

    out_file = FIGURES_DIR / "gradcam_mri_explanations.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM] Saved high-resolution explanation plot to {out_file}")


if __name__ == "__main__":
    run_gradcam_pipeline()
