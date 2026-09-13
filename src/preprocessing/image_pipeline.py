"""
src/preprocessing/image_pipeline.py
=====================================
Image augmentation and DataLoader pipeline for the Brain MRI dataset (SIH26139).

Dataset structure expected:
    data/raw/brain_mri/
        Training/
            glioma/          *.jpg
            meningioma/      *.jpg
            notumor/         *.jpg
            pituitary/       *.jpg
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/

Transform strategy
------------------
TRAIN (augmentation on):
    RandomRotation(15°) → ColorJitter → RandomHorizontalFlip
    → Resize(224, 224) → ToTensor → Normalize(ImageNet stats)

VAL / TEST (deterministic):
    Resize(224, 224) → ToTensor → Normalize(ImageNet stats)

ImageNet normalization statistics:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

Usage
-----
    from src.preprocessing.image_pipeline import get_dataloaders
    loaders = get_dataloaders(batch_size=32)
    for images, labels in loaders['train']:
        ...   # images: [B, 3, 224, 224], labels: [B]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2  # modern transforms API

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
MRI_DIR = ROOT / "data" / "raw" / "brain_mri"

# ─── ImageNet normalisation constants ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# ─── Class mapping ────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ═══════════════════════════════════════════════════════════════════════════
# Transform builders
# ═══════════════════════════════════════════════════════════════════════════

def get_train_transform() -> transforms.Compose:
    """
    Augmented transform for training split.
    Augmentation operations are applied ONLY here to prevent data leakage
    into validation / test sets.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Deterministic transform for validation and test splits.
    No random operations — ensures reproducible evaluation metrics.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Dataset discovery
# ═══════════════════════════════════════════════════════════════════════════

def _find_dataset_root() -> tuple[Path | None, Path | None]:
    """
    Locate Training/ and Testing/ sub-directories in the MRI dataset.
    The Kaggle download may place them directly or inside a nested folder.
    Returns (train_dir, test_dir) or (None, None) if not found.
    """
    # Standard location after Kaggle unzip
    train_dir = MRI_DIR / "Training"
    test_dir  = MRI_DIR / "Testing"

    if train_dir.exists() and test_dir.exists():
        return train_dir, test_dir

    # Search one level deeper (some Kaggle datasets add a wrapper folder)
    for sub in MRI_DIR.iterdir():
        if sub.is_dir():
            candidate_train = sub / "Training"
            candidate_test  = sub / "Testing"
            if candidate_train.exists() and candidate_test.exists():
                return candidate_train, candidate_test

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Main DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    batch_size: int = 32,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for the Brain MRI dataset.

    The Kaggle dataset provides a fixed Training/ and Testing/ split.
    We further split Training/ into train (85%) + val (15%) subsets,
    applying augmentation transforms only to the train subset.

    Args:
        batch_size:    Samples per batch.
        val_fraction:  Fraction of Training/ data used for validation.
        num_workers:   DataLoader worker threads (0 = main process).
        seed:          Random seed for reproducible splits.

    Returns:
        dict with keys 'train', 'val', 'test', each a DataLoader.

    Raises:
        FileNotFoundError: If the MRI dataset directories are not found.
    """
    train_dir, test_dir = _find_dataset_root()

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Brain MRI dataset not found under {MRI_DIR}.\n"
            "Expected structure: data/raw/brain_mri/Training/<class>/ and Testing/<class>/\n"
            "Run `python -m src.preprocessing.ingest --source mri` first."
        )

    # ── Load full Training dataset (with eval transform for split stability) ──
    full_train_ds = datasets.ImageFolder(str(train_dir), transform=get_eval_transform())
    n_val = int(len(full_train_ds) * val_fraction)
    n_train = len(full_train_ds) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train_ds, [n_train, n_val], generator=generator)

    # ── Re-wrap train_subset with augmentation transform ──────────────────────
    # We create a thin wrapper so only the train indices see augmentation
    class _AugSubset(torch.utils.data.Dataset):
        def __init__(self, subset: Subset, transform: transforms.Compose):
            self.subset    = subset
            self.transform = transform

        def __len__(self) -> int:
            return len(self.subset)

        def __getitem__(self, idx: int):
            img, label = self.subset[idx]
            # img is already a tensor from eval_transform; undo normalize & re-augment
            # Simpler: re-load from underlying ImageFolder with augmentation transform
            original_idx = self.subset.indices[idx]
            path, label = self.subset.dataset.samples[original_idx]
            from PIL import Image  # noqa: PLC0415
            img_pil = Image.open(path).convert("RGB")
            return self.transform(img_pil), label

    train_dataset = _AugSubset(train_subset, get_train_transform())
    val_dataset   = val_subset  # already has eval_transform

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_dataset = datasets.ImageFolder(str(test_dir), transform=get_eval_transform())

    # ── DataLoaders ───────────────────────────────────────────────────────────
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    loaders = {
        "train": DataLoader(train_dataset, shuffle=True,  **common_kwargs),
        "val":   DataLoader(val_dataset,   shuffle=False, **common_kwargs),
        "test":  DataLoader(test_dataset,  shuffle=False, **common_kwargs),
    }

    print(
        f"[ImagePipeline] Dataset: {MRI_DIR.name}  "
        f"| train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}"
    )
    print(f"[ImagePipeline] Classes: {full_train_ds.classes}")
    print(f"[ImagePipeline] Batch size: {batch_size}  |  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    return loaders


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loaders = get_dataloaders(batch_size=8)
    imgs, lbls = next(iter(loaders["train"]))
    print(f"Train batch shape:  {imgs.shape}   labels: {lbls}")
    imgs_v, lbls_v = next(iter(loaders["val"]))
    print(f"Val batch shape:    {imgs_v.shape}  labels: {lbls_v}")
    imgs_t, lbls_t = next(iter(loaders["test"]))
    print(f"Test batch shape:   {imgs_t.shape}  labels: {lbls_t}")
