"""
reweight.py — Resampling the train set to break the label–concept correlation.

group_weights      — balanced over discrete (target × concept) groups.  With
                     concept labels from CLIP zero-shot prediction this is a
                     label-free baseline for worst-group accuracy; with true
                     concept labels it is the standard oracle reference.
propensity_weights — inverse propensity on continuous concept scores:
                     w = 1 / p(y | z), with p(y | z) a logistic model on the
                     whitened CLIP concept scores.  Under these weights y is
                     (to the model's accuracy) independent of z, with no
                     thresholding into groups, and z can hold any number of
                     concepts.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, WeightedRandomSampler


def group_weights(labels: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
    """Per-example weights drawing each (label, concept) group equally often."""
    groups = labels.long() * 2 + concepts.long()
    counts = torch.bincount(groups, minlength=4)
    print(
        "  group-balanced sampling — group sizes (y, c): "
        + "  ".join(f"({g // 2},{g % 2})={int(n)}" for g, n in enumerate(counts))
    )
    return (1.0 / counts.clamp_min(1).double())[groups]


def _weighted_corr(y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    yc = y - (w * y).sum()
    zc = z - (w[:, None] * z).sum(0)
    cov = (w[:, None] * yc[:, None] * zc).sum(0)
    return cov / np.sqrt((w * yc**2).sum() * (w[:, None] * zc**2).sum(0))


def propensity_weights(
    labels: torch.Tensor,
    scores: torch.Tensor,
    max_ratio: float = 50.0,
    class_balanced: bool = True,
) -> torch.Tensor:
    """w_i = 1 / p(y_i | z_i)  (or p(y_i) / p(y_i | z_i) if not class_balanced),
    capped at `max_ratio` × the median weight.

    Both make y independent of z; the class-balanced form also gives every
    class equal total mass, as group balancing does.  On Waterbirds (sentence
    prompts) it puts 0.166 / 0.185 of the mass on the two minority groups, vs
    0.150 / 0.178 for CLIP pseudo-group balancing, without forming groups.

    labels : (N,) task labels;  scores : (N, r) concept scores (e.g. whitened
    CLIP scores from clip_oracle.whitened_concept_scores) — no concept labels.
    """
    y = labels.long().numpy()
    z = scores.double().numpy()
    model = LogisticRegression(max_iter=2000).fit(z, y)
    p_y_given_z = model.predict_proba(z)[
        np.arange(len(y)), np.searchsorted(model.classes_, y)
    ]
    w = 1.0 / np.clip(p_y_given_z, 1e-6, None)
    if not class_balanced:
        w *= np.bincount(y)[y] / len(y)
    cap = max_ratio * np.median(w)
    n_capped = int((w > cap).sum())
    w = np.minimum(w, cap)
    ess = w.sum() ** 2 / (w**2).sum() / len(w)
    before = _weighted_corr(y.astype(float), z, np.ones_like(w))
    after = _weighted_corr(y.astype(float), z, w)
    print(
        f"  propensity sampling — p(y|z) model train acc {model.score(z, y):.3f}  "
        f"effective sample size {ess:.1%}  capped {n_capped}"
    )
    print(
        "    corr(y, z_k) before: "
        + " ".join(f"{v:+.3f}" for v in before)
        + "   after: "
        + " ".join(f"{v:+.3f}" for v in after)
    )
    return torch.from_numpy(w)


def weighted_loader(loader: DataLoader, weights: torch.Tensor, seed: int) -> DataLoader:
    """Same dataset/batch size as `loader`, drawn with probability ∝ `weights`
    (aligned with `loader.dataset` indices).  One epoch = len(dataset) draws
    with replacement."""
    sampler = WeightedRandomSampler(
        weights.double(),
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


def group_balanced_loader(
    loader: DataLoader,
    labels: torch.Tensor,
    concepts: torch.Tensor,
    seed: int,
) -> DataLoader:
    """weighted_loader with group_weights (kept for existing callers)."""
    return weighted_loader(loader, group_weights(labels, concepts), seed)
