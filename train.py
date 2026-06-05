#!/usr/bin/env python
"""
train.py — Multi-seed ablation grid: ERM baseline + debiasing variants.

Runs all five conditions across N seeds and writes a results summary to
results/summary.csv for easy analysis.

Usage
-----
    # Full ablation grid, 3 seeds
    python train.py

    # Quick smoke-test: one condition, one seed, 2 epochs
    python train.py --runs erm full --seeds 42 --epochs 2

    # Single ERM baseline
    python train.py --runs erm --seeds 42 123 456
"""

import argparse
import csv
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from clip_debias.clip_oracle import CLIPOracle, build_concept_direction
from clip_debias.config import (
    erm_config,
    align_only_config,
    repulse_only_config,
    full_config,
    full_strong_config,
)
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.trainer import Trainer

# ── All available run configurations ─────────────────────────────────────────
# Ordered so ERM always runs first (natural reference point).
RUN_FACTORIES = {
    "erm": erm_config,
    "align_only": align_only_config,
    "repulse_only": repulse_only_config,
    "full": full_config,
    "full_strong": full_strong_config,
}

SEEDS = [42, 123, 456]


# ── Utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed ablation grid for CLIP-guided debiasing"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUN_FACTORIES),
        default=list(RUN_FACTORIES),
        help="Which run configurations to execute (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds to average over",
    )
    parser.add_argument("--celeba_root", default="./data/celeba")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


# ── CSV logging ───────────────────────────────────────────────────────────────


def append_result(path: str, row: dict):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_summary(all_results: List[dict]):
    """Print mean ± std for each run name across seeds."""
    from collections import defaultdict

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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "summary.csv")

    # ── Shared setup (done once, reused across runs) ───────────────────────────
    # We build a reference config just to get dataloaders and CLIP.
    ref_cfg = erm_config(
        celeba_root=args.celeba_root,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        amp=not args.no_amp,
        device=device,
    )

    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(ref_cfg)
    print(
        f"  Train: {len(train_loader.dataset):,}  "
        f"Val: {len(val_loader.dataset):,}  "
        f"Test: {len(test_loader.dataset):,}"
    )

    # Load CLIP once — it's frozen and shared across all seeds and runs.
    # Skip if every requested run is ERM (CLIP not needed).
    need_clip = any(r != "erm" for r in args.runs)
    oracle = v_t_hat = None
    if need_clip:
        print(f"\nLoading CLIP ({ref_cfg.clip_model}) …")
        oracle = CLIPOracle(ref_cfg.clip_model, device=device)
        print("Computing concept direction …")
        v_t_hat = build_concept_direction(
            oracle,
            prompts_pos=ref_cfg.concept_prompts_pos,
            prompts_neg=ref_cfg.concept_prompts_neg,
        )
        print(f"  ‖v_t_hat‖ = {v_t_hat.norm().item():.6f}  (should be 1.0)")

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

            # Build a fresh config for this (run, seed) pair
            cfg = RUN_FACTORIES[run_name](
                celeba_root=args.celeba_root,
                backbone=args.backbone,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                checkpoint_dir=args.checkpoint_dir,
                amp=not args.no_amp,
                device=device,
                seed=seed,
            )

            # Fresh model with same seed → reproducible initialisation
            model = build_model(cfg, num_classes=2)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_params:,}")

            # Train
            is_erm = run_name == "erm"
            trainer = Trainer(
                model=model,
                cfg=cfg,
                oracle=None if is_erm else oracle,
                v_t_hat=None if is_erm else v_t_hat,
            )
            best_ckpt = trainer.fit(train_loader, val_loader)

            # Load best checkpoint for evaluation
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

            # Evaluate
            metrics = run_evaluation(
                model,
                train_loader,
                test_loader,
                device,
                cfg.amp,
                label=f"{run_name}  seed={seed}",
            )

            # Log
            row = {"run_name": run_name, "seed": seed, **metrics}
            all_results.append(row)
            append_result(csv_path, row)
            print(f"\n  → Results written to {csv_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print_summary(all_results)
    print(f"Full per-run results saved to: {csv_path}")


if __name__ == "__main__":
    main()
