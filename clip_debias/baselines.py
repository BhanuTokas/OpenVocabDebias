"""
baselines.py — Group-robustness baselines trained with the same data pipeline,
backbone, schedule and label-free checkpoint selection as the main method.

  JTT  (Liu et al., ICML 2021)   train errors of an early ERM checkpoint are
        upsampled by `upweight` (--error_set, --upweight); error sets come
        from error_set.py.
  BAM  (Li et al., TMLR 2024)    stage 1: logits f(x_i) + λ·b_i with a learned
        per-example auxiliary vector b_i (zeros at init, trained jointly; --bam_aux λ);
        its early-checkpoint errors (auxiliaries removed) form the error set,
        and stage 2 is JTT-style upsampling.
  GroupDRO (Sagawa et al., ICLR 2020)  exponentiated-gradient weights over
        (target × concept) groups (--group_dro pseudo | oracle, --dro_eta);
        with CLIP zero-shot pseudo-groups this is the mitigation step of
        B2T / ViG-Bias, with true groups the usual upper reference.

DFR and AFR (last-layer retraining) and the frozen-CLIP probe are post-hoc and
live in last_layer.py and clip_probe.py.

The loss terms that need to know which examples are in a batch get the sample
indices from IndexedDataset, which replaces the (unused in training) concept
label with the dataset index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class IndexedDataset(Dataset):
    """(image, label, index) instead of (image, label, concept)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label, _ = self.base[idx]
        return image, label, idx


def indexed_loader(loader: DataLoader) -> DataLoader:
    """Same batching as `loader` (shuffled), over IndexedDataset(loader.dataset)."""
    return DataLoader(
        IndexedDataset(loader.dataset),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )


class GroupDROLoss(nn.Module):
    """Online GroupDRO (Sagawa et al. 2020, Algorithm 1): q_g ← q_g·exp(η·L_g),
    renormalised after every batch; loss = Σ_g q_g L_g with L_g the mean loss of
    group g in the batch (groups absent from the batch keep their weight)."""

    def __init__(self, n_groups: int, eta: float):
        super().__init__()
        self.eta = eta
        self.register_buffer("q", torch.full((n_groups,), 1.0 / n_groups))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor):
        losses = F.cross_entropy(logits.float(), labels, reduction="none")
        onehot = F.one_hot(groups, self.q.numel()).float()  # (B, G)
        counts = onehot.sum(0)
        group_loss = (onehot.T @ losses) / counts.clamp_min(1.0)
        with torch.no_grad():
            present = counts > 0
            self.q[present] *= torch.exp(self.eta * group_loss[present].detach())
            self.q /= self.q.sum()
        return (self.q * group_loss).sum()


def fold_indices(n: int, n_folds: int, fold: int, seed: int) -> np.ndarray:
    """Train indices held out in `fold` of a seeded random K-fold split of n."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(np.array_split(perm, n_folds)[fold])


def fold_loader(loader: DataLoader, n_folds: int, fold: int, seed: int) -> DataLoader:
    """Same batching as `loader` (shuffled), over the train examples outside
    `fold` (fold_indices)."""
    held = fold_indices(len(loader.dataset), n_folds, fold, seed)
    keep = np.setdiff1d(np.arange(len(loader.dataset)), held)
    print(
        f"  fold {fold}/{n_folds} (split seed {seed}): training on {len(keep):,}, "
        f"holding out {len(held):,} train examples"
    )
    return DataLoader(
        Subset(loader.dataset, keep.tolist()),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )


def error_set_weights(path: str, upweight: float) -> torch.Tensor:
    """Sampling weights for JTT / BAM stage 2: `upweight` for the train examples in
    the error set (error_set.py file), 1 elsewhere; aligned with train indices."""
    err = np.load(path)["err"].astype(bool)
    print(
        f"  error set {path}: {err.sum():,} / {len(err):,} train examples "
        f"upweighted x{upweight:g}"
    )
    return torch.from_numpy(np.where(err, upweight, 1.0))


def add_baseline_args(parser) -> None:
    g = parser.add_argument_group("baselines (clip_debias/baselines.py)")
    g.add_argument(
        "--group_dro",
        default="none",
        choices=["none", "pseudo", "oracle"],
        help="GroupDRO over (target x concept) groups from CLIP zero-shot prompts "
        "('pseudo', label-free; B2T-style) or the dataset ('oracle', reference). "
        "Adds '_dro-<mode>'.",
    )
    g.add_argument("--dro_eta", type=float, default=0.01, help="GroupDRO step size")
    g.add_argument(
        "--error_set",
        default=None,
        help="JTT / BAM stage 2: npz from error_set.py; its train errors are upsampled "
        "x --upweight. Adds '_up<U>-<file stem>'.",
    )
    g.add_argument("--upweight", type=float, default=50.0)
    g.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="AdamW weight decay (default: the config's 1e-4). Adds '_wd<v>'; "
        "strong values (e.g. 1.0 on Waterbirds) are the usual GroupDRO setting.",
    )
    g.add_argument(
        "--bam_aux",
        type=float,
        default=0.0,
        help="BAM stage 1: coefficient λ of the per-example auxiliary logits. "
        "Adds '_bam<λ>'.",
    )
    g.add_argument(
        "--n_folds",
        type=int,
        default=0,
        help="cross-fitting for discover_biases.py: train on all folds of a "
        "--fold_seed K-fold split of train except --fold. Adds '_f<k>of<K>s<seed>'.",
    )
    g.add_argument("--fold", type=int, default=0)
    g.add_argument("--fold_seed", type=int, default=0)


def apply_baseline_args(cfg, args):
    """Record the baseline settings in cfg and extend its run name."""
    if args.group_dro != "none":
        cfg.group_dro, cfg.dro_eta = args.group_dro, args.dro_eta
        cfg.run_name += f"_dro-{args.group_dro}"
        if args.dro_eta != 0.01:
            cfg.run_name += f"_eta{args.dro_eta:g}"
    if args.error_set:
        cfg.error_set, cfg.upweight = args.error_set, args.upweight
        cfg.run_name += f"_up{args.upweight:g}-{Path(args.error_set).stem}"
    if args.bam_aux > 0:
        cfg.bam_aux = args.bam_aux
        cfg.run_name += f"_bam{args.bam_aux:g}"
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
        cfg.run_name += f"_wd{args.weight_decay:g}"
    if args.n_folds > 0:
        if not 0 <= args.fold < args.n_folds:
            raise SystemExit(f"--fold must be in [0, {args.n_folds})")
        cfg.n_folds, cfg.fold, cfg.fold_seed = args.n_folds, args.fold, args.fold_seed
        cfg.run_name += f"_f{args.fold}of{args.n_folds}s{args.fold_seed}"
    return cfg


def dro_groups(
    args, y_tr: torch.Tensor, c_tr: torch.Tensor, c_hat: torch.Tensor | None
) -> torch.Tensor | None:
    """Per-train-example group ids y·G + c for --group_dro, where G is the number
    of concept codes (2^attributes)."""
    if args.group_dro == "none":
        return None
    c = c_hat if args.group_dro == "pseudo" else c_tr
    n_codes = int(max(c_tr.max().item(), c.max().item())) + 1
    n_codes = 1 << max(1, (n_codes - 1).bit_length())
    groups = y_tr.long() * n_codes + c.long()
    counts = torch.bincount(groups, minlength=2 * n_codes)
    print(f"  GroupDRO ({args.group_dro}) train group sizes: {counts.tolist()}")
    return groups
