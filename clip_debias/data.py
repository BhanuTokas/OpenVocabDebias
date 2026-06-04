"""
data.py — CelebA data loading with separate splits for train / val / test.

CelebA attribute indices (torchvision):
  Each sample returns (image_tensor, attr_tensor) where attr_tensor is a
  40-element binary vector.  We expose helpers that pull out the target
  label and the concept label for downstream eval.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA

# CelebA attribute names in torchvision order
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

_ATTR_TO_IDX = {name: i for i, name in enumerate(CELEBA_ATTRS)}


def attr_index(name: str) -> int:
    """Return the integer index for a CelebA attribute name."""
    if name not in _ATTR_TO_IDX:
        raise ValueError(f"Unknown CelebA attribute '{name}'. "
                         f"Available: {list(_ATTR_TO_IDX)}")
    return _ATTR_TO_IDX[name]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Wrapper Dataset ───────────────────────────────────────────────────────────

class CelebADebias(Dataset):
    """
    Thin wrapper around torchvision CelebA that returns
        (image, target_label, concept_label)
    where both labels are scalar {0, 1} tensors.
    """

    def __init__(
        self,
        root: str,
        split: str,          # "train" | "valid" | "test"
        target_attr: str,
        concept_attr: str,
        transform=None,
    ):
        self.base = CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=transform,
            download=False,   # set True on first run
        )
        self.target_idx = attr_index(target_attr)
        self.concept_idx = attr_index(concept_attr)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, attrs = self.base[idx]
        target = attrs[self.target_idx].long()
        concept = attrs[self.concept_idx].long()
        return image, target, concept


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) using settings from cfg.
    """
    train_ds = CelebADebias(
        cfg.celeba_root, "train",
        cfg.target_attr, cfg.concept_attr,
        transform=get_train_transform(),
    )
    val_ds = CelebADebias(
        cfg.celeba_root, "valid",
        cfg.target_attr, cfg.concept_attr,
        transform=get_eval_transform(),
    )
    test_ds = CelebADebias(
        cfg.celeba_root, "test",
        cfg.target_attr, cfg.concept_attr,
        transform=get_eval_transform(),
    )

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
