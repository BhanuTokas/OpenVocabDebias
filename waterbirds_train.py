#!/usr/bin/env python
"""
waterbirds_train.py — Multi-seed ablation grid on Waterbirds.

Mirrors train.py exactly.  The only differences are:
  1. Data   : WaterbirdsDebiasDataset (grodino/waterbirds) instead of CelebA
  2. Concept: water-related words vs land/vegetation words as prompt poles
  3. Paths  : checkpoint_dir and results_dir default to waterbirds-specific dirs

Usage
-----
    # Full ablation grid, 3 seeds
    python waterbirds_train.py

    # Quick smoke-test: one condition, one seed, 2 epochs
    python waterbirds_train.py --runs erm full --seeds 42 --epochs 2

    # Single ERM baseline
    python waterbirds_train.py --runs erm --seeds 42 123 456
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clip_debias.clip_oracle import CLIPOracle, compute_concept_subspace
from clip_debias.config import (
    DebiasingConfig,
    align_only_config,
    erm_config,
    full_config,
    full_strong_config,
    repulse_only_config,
)
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.trainer import Trainer
from clip_debias.waterbirds_data import build_waterbirds_dataloaders

# ── Waterbirds-specific config defaults ───────────────────────────────────────


@dataclass
class WaterbirdsConfig(DebiasingConfig):
    concept_attr: str = "place"
    checkpoint_dir: str = "./checkpoints_waterbirds"


RUN_FACTORIES = {
    "erm": erm_config,
    "align_only": align_only_config,
    "repulse_only": repulse_only_config,
    "full": full_config,
    "full_strong": full_strong_config,
}

SEEDS = [42, 123, 456]

POS_WORDS = ["ocean", "beach", "shore", "water", "waves"]
NEG_WORDS = [
    "forest",
    "foliage",
    "tree",
    "stalks",
    "branch",
    "trees",
    "branches",
    "vegetation",
]


# ── Utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_result(path: str, row: dict):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_summary(all_results: List[dict]):
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in all_results:
        grouped[r["run_name"]].append(r)

    metric_keys = ["probe_test_acc", "task_test_acc", "worst_group_acc"]
    header = f"{'run':<16}" + "".join(f"  {k:<26}" for k in metric_keys)
    print(f"\n{'='*80}")
    print("  Final summary (mean ± std across seeds)")
    print(f"{'='*80}")
    print(header)

    for run_name, rows in grouped.items():
        line = f"  {run_name:<14}"
        for k in metric_keys:
            vals = [r[k] for r in rows if k in r]
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                line += f"  {mean:.4f} ± {std:.4f}          "
            else:
                line += f"  {'N/A':<26}"
        print(line)
    print()


def biased_subset(loader, n: int, bias: float, seed: int = 42) -> DataLoader:
    """
    Sample n examples with a controlled spurious correlation.
    bias=0.95 → 95/5, bias=0.50 → 50/50 within each class.
    """
    ds = loader.dataset
    hf = ds.ds

    groups: dict = {(l, p): [] for l in [0, 1] for p in [0, 1]}
    for i, (l, p) in enumerate(zip(hf["label"], hf["place"])):
        groups[(l, p)].append(i)

    rng = random.Random(seed)
    n_per_class = n // 2
    n_aligned = int(n_per_class * bias)
    n_misaligned = n_per_class - n_aligned

    selected = []
    for label, ap, mp in [(0, 0, 1), (1, 1, 0)]:
        selected.extend(
            rng.sample(groups[(label, ap)], min(n_aligned, len(groups[(label, ap)])))
        )
        selected.extend(
            rng.sample(groups[(label, mp)], min(n_misaligned, len(groups[(label, mp)])))
        )

    return DataLoader(
        Subset(ds, selected),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed ablation grid for Waterbirds CLIP debiasing"
    )
    parser.add_argument(
        "--runs", nargs="+", choices=list(RUN_FACTORIES), default=list(RUN_FACTORIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "vit_b_16"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="./checkpoints_waterbirds")
    parser.add_argument("--results_dir", default="./results_waterbirds")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small subset (128 train / 32 val / 32 test) for quick testing",
    )
    parser.add_argument(
        "--concept_subspace_k",
        type=int,
        default=3,
        help="Number of SVD components for concept subspace (default: 3; "
        "use 1 to reproduce single-direction behaviour)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "summary.csv")

    ref_cfg = WaterbirdsConfig(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        amp=not args.no_amp,
        device=device,
        concept_subspace_k=args.concept_subspace_k,
    )

    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_waterbirds_dataloaders(ref_cfg)
    if args.quick:
        train_loader = biased_subset(train_loader, 128, bias=0.95)
        val_loader = biased_subset(val_loader, 32, bias=0.50)
        test_loader = biased_subset(test_loader, 32, bias=0.50)
    print(
        f"  Train: {len(train_loader.dataset):,}  "
        f"Val: {len(val_loader.dataset):,}  "
        f"Test: {len(test_loader.dataset):,}"
    )

    # ── Load CLIP and compute concept subspace (once, shared across all runs) ─
    need_clip = any(r != "erm" for r in args.runs)
    oracle = subspace = None
    if need_clip:
        print(f"\nLoading CLIP ({ref_cfg.clip_model}) …")
        oracle = CLIPOracle(ref_cfg.clip_model, device=device)

        # pos_words = get_synonyms("water")
        # neg_words = get_random_base_nouns(n=len(pos_words)*5, exclude_words=set(pos_words), seed=42)
        print(f"  pos: {POS_WORDS}")
        print(f"  neg: {NEG_WORDS}")

        print(f"Computing concept subspace (k={args.concept_subspace_k}) …")
        subspace = compute_concept_subspace(
            oracle,
            prompts_pos=POS_WORDS,
            prompts_neg=NEG_WORDS,
            k=args.concept_subspace_k,
        )
        print(f"  subspace shape: {tuple(subspace.shape)}")

    # ── Ablation grid ─────────────────────────────────────────────────────────
    all_results: List[dict] = []
    total_runs = len(args.runs) * len(args.seeds)
    run_idx = 0

    for run_name in args.runs:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'#'*70}")
            print(f"  Run {run_idx}/{total_runs}: {run_name}  seed={seed}")
            print(f"{'#'*70}")

            set_seed(seed)

            cfg = RUN_FACTORIES[run_name](
                backbone=args.backbone,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                checkpoint_dir=args.checkpoint_dir,
                amp=not args.no_amp,
                device=device,
                seed=seed,
                concept_subspace_k=args.concept_subspace_k,
            )
            cfg.concept_attr = "place"

            model = build_model(cfg, num_classes=2)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_params:,}")

            is_erm = run_name == "erm"
            trainer = Trainer(
                model=model,
                cfg=cfg,
                oracle=None if is_erm else oracle,
                subspace=None if is_erm else subspace,
            )
            best_ckpt = trainer.fit(train_loader, val_loader)

            model.load_state_dict(torch.load(best_ckpt, map_location=device))

            metrics = run_evaluation(
                model,
                train_loader,
                test_loader,
                device,
                cfg.amp,
                label=f"{run_name}  seed={seed}",
            )

            row = {
                "run_name": run_name,
                "seed": seed,
                "subspace_k": args.concept_subspace_k,
                **metrics,
            }
            all_results.append(row)
            append_result(csv_path, row)
            print(f"\n  → Results written to {csv_path}")

    print_summary(all_results)
    print(f"Full per-run results saved to: {csv_path}")


if __name__ == "__main__":
    main()
