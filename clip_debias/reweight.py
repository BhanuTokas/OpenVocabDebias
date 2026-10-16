"""
reweight.py — Group-balanced resampling over (target × concept) groups.

With concept labels from CLIP zero-shot prediction this is a label-free
baseline for worst-group accuracy (no concept annotations needed); with true
concept labels it is the standard oracle group-reweighting reference.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def group_balanced_loader(
    loader: DataLoader,
    labels: torch.Tensor,
    concepts: torch.Tensor,
    seed: int,
) -> DataLoader:
    """Same dataset/batch size as `loader`, sampled so each (label, concept)
    group is drawn equally often.  `labels`/`concepts` must be aligned with
    `loader.dataset` indices.  One epoch = len(dataset) draws with replacement.
    """
    groups = labels.long() * 2 + concepts.long()
    counts = torch.bincount(groups, minlength=4)
    weights = (1.0 / counts.clamp_min(1).double())[groups]
    print(
        "  group-balanced sampling — group sizes (y, c): "
        + "  ".join(f"({g // 2},{g % 2})={int(n)}" for g, n in enumerate(counts))
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )
