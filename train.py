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
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_concept_subspace,
    embed_dataset,
    fit_leace_eraser,
    zeroshot_concept_labels,
)
from clip_debias.config import (
    erm_config,
    align_only_config,
    repulse_only_config,
    full_config,
    full_strong_config,
    sequential_config,
    disable_lambda_schedules,
    use_scratch_backbone,
    use_leace_eraser,
    use_whitened_repulse,
    use_group_reweighting,
    get_concept_prompts,
)
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.results_io import append_result
from clip_debias.reweight import group_balanced_loader
from clip_debias.trainer import Trainer

RUN_FACTORIES = {
    "erm": erm_config,
    "align_only": align_only_config,
    "repulse_only": repulse_only_config,
    "full": full_config,
    "full_strong": full_strong_config,
    "sequential": sequential_config,
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
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--celeba_root", default="./data/celeba")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "vit_b_16"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers; CelebA JPEG decoding is the bottleneck, so match "
        "the allocated CPU cores",
    )
    parser.add_argument(
        "--target_attr",
        default="Attractive",
        help="CelebA attribute to classify (e.g. Attractive, Smiling, Blond_Hair)",
    )
    parser.add_argument(
        "--concept_attr",
        default="Male",
        help="CelebA attribute to debias against (e.g. Male, Young, Eyeglasses)",
    )
    parser.add_argument(
        "--concept_prompts_pos",
        nargs="+",
        default=None,
        help="Override positive concept prompts (auto-selected from library if omitted)",
    )
    parser.add_argument(
        "--concept_prompts_neg",
        nargs="+",
        default=None,
        help="Override negative concept prompts (auto-selected from library if omitted)",
    )
    parser.add_argument(
        "--concept_subspace_k",
        type=int,
        default=3,
        help="Number of SVD components for concept subspace (default: 3; use 1 to "
        "reproduce single-direction behaviour)",
    )
    parser.add_argument(
        "--sequential_switch_epoch",
        type=int,
        default=5,
        help="For the 'sequential' run: epoch at which training hard-switches "
        "from repulse-only to align-only (default: 5, for 10-epoch runs)",
    )
    parser.add_argument(
        "--no_lambda_schedules",
        action="store_true",
        help="Use constant lambdas: disable the calibrated lambda_task/lambda_align "
        "warmup and lambda_repulse decay. Affected runs get a '_nosched' suffix.",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Randomly initialise the backbone instead of loading ImageNet weights. "
        "All runs get a '_scratch' suffix; raise --epochs/--lr accordingly.",
    )
    parser.add_argument(
        "--eraser",
        default="orth",
        choices=["orth", "leace"],
        help="How V_I_perp is formed: orthogonal projection of the text subspace, "
        "or whitened LEACE-style erasure fit on the train-set CLIP image covariance "
        "(label-free). Runs using the alignment target get a '_leace' suffix.",
    )
    parser.add_argument(
        "--repulse",
        default="orth",
        choices=["orth", "whitened", "whitened_cond"],
        help="What L_repulse penalises: P(E(x)) along the text-subspace basis, or "
        "the batch correlation between E(x) and the whitened CLIP concept scores "
        "(label-free); 'whitened_cond' computes that correlation within each class. "
        "Runs with lambda_repulse > 0 get a '_wrep' / '_wcrep' suffix.",
    )
    parser.add_argument(
        "--lambda_repulse",
        type=float,
        default=None,
        help="Override lambda_repulse for runs that have a repulse term "
        "(adds a '_lrep<value>' suffix). Default: each run config's own value.",
    )
    parser.add_argument(
        "--group_reweight",
        default="none",
        choices=["none", "pseudo", "oracle"],
        help="Group-balanced resampling over (target x concept) groups, with concept "
        "labels from CLIP zero-shot prediction ('pseudo', label-free) or the "
        "dataset ('oracle', reference only). Adds a '_grw_<mode>' suffix.",
    )
    return parser.parse_args()


def print_summary(all_results: List[dict]):
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

    # ── Resolve concept prompts ───────────────────────────────────────────────
    if args.concept_prompts_pos is not None or args.concept_prompts_neg is not None:
        prompts_pos, prompts_neg = get_concept_prompts(args.concept_attr)
        prompts_pos = args.concept_prompts_pos or prompts_pos
        prompts_neg = args.concept_prompts_neg or prompts_neg
        print(f"Using custom concept prompts for '{args.concept_attr}'")
    else:
        prompts_pos, prompts_neg = get_concept_prompts(args.concept_attr)
        print(f"Auto-selected prompts for concept '{args.concept_attr}'")
    print(f"  pos: {prompts_pos}")
    print(f"  neg: {prompts_neg}")

    shared_overrides = dict(
        celeba_root=args.celeba_root,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        device=device,
        target_attr=args.target_attr,
        concept_attr=args.concept_attr,
        concept_prompts_pos=prompts_pos,
        concept_prompts_neg=prompts_neg,
        concept_subspace_k=args.concept_subspace_k,
        sequential_phase_switch_epoch=args.sequential_switch_epoch,
    )

    ref_cfg = erm_config(**shared_overrides)

    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(ref_cfg)
    print(
        f"  Train: {len(train_loader.dataset):,}  "
        f"Val: {len(val_loader.dataset):,}  "
        f"Test: {len(test_loader.dataset):,}"
    )

    # ── Load CLIP and compute concept subspace (once, shared across all runs) ─
    # CLIP is always loaded: val pseudo-labels drive best_pseudo_wga.pt for every run
    print(f"\nLoading CLIP ({ref_cfg.clip_model}) …")
    oracle = CLIPOracle(ref_cfg.clip_model, device=device)
    print(f"Computing concept subspace (k={args.concept_subspace_k}) …")
    subspace = compute_concept_subspace(
        oracle,
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        k=args.concept_subspace_k,
    )
    print(f"  subspace shape: {tuple(subspace.shape)}")

    # ── Val pseudo-labels: label-free checkpoint selection (best_pseudo_wga.pt) ─
    print("\nEmbedding the val split with CLIP …")
    v_va, _, c_va = embed_dataset(
        oracle, val_loader.dataset, num_workers=ref_cfg.num_workers
    )
    val_pseudo = zeroshot_concept_labels(oracle, prompts_pos, prompts_neg, v_va)
    print(
        "  zero-shot concept accuracy on val (reported only, not used): "
        f"{(val_pseudo == c_va).float().mean():.4f}"
    )

    # ── One CLIP pass over train: whitened eraser and/or group labels ─────────
    need_eraser = args.eraser == "leace" or args.repulse != "orth"
    eraser = train_groups = None
    if need_eraser or args.group_reweight != "none":
        print("\nEmbedding the train split with CLIP …")
        v_tr, y_tr, c_tr = embed_dataset(
            oracle, train_loader.dataset, num_workers=ref_cfg.num_workers
        )
        if need_eraser:
            eraser = fit_leace_eraser(v_tr, subspace)
            print(f"  fitted whitened eraser on {len(v_tr):,} train embeddings")
        if args.group_reweight == "pseudo":
            c_hat = zeroshot_concept_labels(oracle, prompts_pos, prompts_neg, v_tr)
            print(
                "  zero-shot concept accuracy on train (reported only, not used): "
                f"{(c_hat == c_tr).float().mean():.4f}"
            )
            train_groups = (y_tr, c_hat)
        elif args.group_reweight == "oracle":
            train_groups = (y_tr, c_tr)

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

            cfg = RUN_FACTORIES[run_name](**shared_overrides, seed=seed)
            if args.no_lambda_schedules:
                cfg = disable_lambda_schedules(cfg)
            if args.scratch:
                cfg = use_scratch_backbone(cfg)
            if args.eraser == "leace":
                cfg = use_leace_eraser(cfg)
            if args.repulse != "orth":
                cfg = use_whitened_repulse(cfg, args.repulse == "whitened_cond")
            if args.lambda_repulse is not None and cfg.lambda_repulse > 0:
                cfg.lambda_repulse = args.lambda_repulse
                cfg.run_name += f"_lrep{args.lambda_repulse:g}"
            if args.group_reweight != "none":
                cfg = use_group_reweighting(cfg, args.group_reweight)
            model = build_model(cfg, num_classes=2)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {n_params:,}")

            is_erm = run_name == "erm"
            trainer = Trainer(
                model=model,
                cfg=cfg,
                oracle=None if is_erm else oracle,
                subspace=None if is_erm else subspace,
                eraser=None if is_erm else eraser,
                val_pseudo_concepts=val_pseudo,
            )
            fit_loader = (
                group_balanced_loader(train_loader, *train_groups, seed=seed)
                if train_groups is not None
                else train_loader
            )
            best_ckpt = trainer.fit(fit_loader, val_loader)

            model.load_state_dict(torch.load(best_ckpt, map_location=device))

            metrics = run_evaluation(
                model,
                train_loader,
                test_loader,
                device,
                cfg.amp,
                label=f"{cfg.run_name}  seed={seed}",
            )

            row = {
                "run_name": cfg.run_name,
                "seed": seed,
                "target_attr": args.target_attr,
                "concept_attr": args.concept_attr,
                "subspace_k": args.concept_subspace_k,
                "run_dir": trainer.ckpt_dir,
                **metrics,
            }
            all_results.append(row)
            append_result(csv_path, row)
            print(f"\n  → Results written to {csv_path}")

    print_summary(all_results)
    print(f"Full per-run results saved to: {csv_path}")


if __name__ == "__main__":
    main()
