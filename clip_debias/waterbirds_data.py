"""
waterbirds_data.py — Waterbirds dataset loading via HuggingFace datasets.

Dataset: grodino/waterbirds
Splits : train (4800), validation (1200), test (5788)
Fields : image (PIL), label (0=landbird/1=waterbird), place (0=land_bg/1=water_bg)
"""

from __future__ import annotations

from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from .data import get_eval_transform, get_train_transform


class WaterbirdsDebiasDataset(Dataset):
    """
    Thin wrapper around grodino/waterbirds that returns
        (image_tensor, label, place)
    where both labels are scalar {0, 1} tensors.

    label : 0=landbird, 1=waterbird  — task label
    place : 0=land_bg,  1=water_bg   — concept label used for probe eval
    """

    def __init__(self, split: str, transform=None):
        # split: "train" | "validation" | "test"
        self.ds = load_dataset("grodino/waterbirds", split=split)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.ds[idx]
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        place = torch.tensor(sample["place"], dtype=torch.long)
        return image, label, place


def build_waterbirds_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    Uses the same transform pipeline as CelebA (ImageNet normalisation).
    """
    train_ds = WaterbirdsDebiasDataset("train", transform=get_train_transform())
    val_ds = WaterbirdsDebiasDataset("validation", transform=get_eval_transform())
    test_ds = WaterbirdsDebiasDataset("test", transform=get_eval_transform())

    common_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kw)

    return train_loader, val_loader, test_loader
