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
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_multi_concept_subspace,
    embed_dataset,
    fit_leace_eraser,
    fit_zeroshot_cutoffs,
    whitened_concept_scores,
    zeroshot_concept_codes,
)
from clip_debias.config import (
    DebiasingConfig,
    align_only_config,
    erm_config,
    full_config,
    full_strong_config,
    sequential_config,
    disable_lambda_schedules,
    use_scratch_backbone,
    use_leace_eraser,
    use_whitened_repulse,
    use_group_reweighting,
    repulse_only_config,
)
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.results_io import append_result
from clip_debias.discovery import load_core_dirs, load_prompt_file
from clip_debias.reweight import group_weights, propensity_weights, weighted_loader
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
    "sequential": sequential_config,
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

# Waterbirds backgrounds come from Places: ocean / natural lake (water) and
# bamboo forest / broadleaf forest (land).  Same length → per-pair differences.
WATER_PLACES = [
    "over the ocean",
    "on a lake",
    "at the beach",
    "near the shore",
    "on the water",
    "by the sea",
]
LAND_PLACES = [
    "in a bamboo forest",
    "in a broadleaf forest",
    "in the woods",
    "on a tree branch",
    "among green foliage",
    "in a forest",
]
WB_TEMPLATES = [
    "a photo of a bird {}",
    "a bird {}",
    "a photo of something {}",
    "a picture taken {}",
]

# Concept prompt sets (pos = water, neg = land).  They define the concept
# subspace, the whitened eraser and the CLIP zero-shot pseudo-labels.
PROMPT_SETS = {
    "words": (POS_WORDS, NEG_WORDS),
    "sentences": (
        [f"a photo of a bird {p}" for p in WATER_PLACES],
        [f"a photo of a bird {p}" for p in LAND_PLACES],
    ),
    "ensemble": (
        [t.format(p) for t in WB_TEMPLATES for p in WATER_PLACES],
        [t.format(p) for t in WB_TEMPLATES for p in LAND_PLACES],
    ),
}


# ── Utilities ─────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        choices=["orth", "whitened", "whitened_cond", "raw"],
        help="What L_repulse penalises: P(E(x)) along the text-subspace basis, or "
        "the batch correlation between E(x) and the whitened CLIP concept scores "
        "(label-free); 'whitened_cond' computes that correlation within each class; "
        "'raw' uses the centred raw text-subspace scores instead (ablation). "
        "Runs with lambda_repulse > 0 get a '_wrep' / '_wcrep' / '_rrep' suffix.",
    )
    parser.add_argument(
        "--align",
        default="proj",
        choices=["proj", "cka"],
        help="What L_align does: cosine of the projection head to V_I_perp, or "
        "1 - linear CKA between E(x) and the LEACE-erased CLIP embedding (no head). "
        "'cka' adds an '_acka' suffix to runs with lambda_align > 0.",
    )
    parser.add_argument(
        "--lambda_align",
        type=float,
        default=None,
        help="Override lambda_align for runs that have an align term "
        "(adds a '_lal<value>' suffix). Default: each run config's own value.",
    )
    parser.add_argument(
        "--eraser_fit",
        default="biased",
        choices=["biased", "balanced"],
        help="Fit the whitened eraser / concept scores on the train set as is, or "
        "reweighted to balance the (target x CLIP zero-shot concept) pseudo-groups "
        "(label-free). 'balanced' adds an '_efbal' suffix.",
    )
    parser.add_argument(
        "--zs_cutoff",
        default="zero",
        choices=["zero", "gmm"],
        help="Cutoff of the CLIP zero-shot concept labels (pseudo-groups and the "
        "val pseudo-groups for best_pseudo_wga.pt): 0, or fitted per attribute on "
        "the unlabeled train scores with a 2-component Gaussian mixture. 'gmm' "
        "adds a '_zsgmm' suffix to every run.",
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
        choices=["none", "pseudo", "oracle", "propensity"],
        help="Group-balanced resampling over (target x concept) groups, with concept "
        "labels from CLIP zero-shot prediction ('pseudo', label-free) or the "
        "dataset ('oracle', reference only); 'propensity' weights by "
        "1/p(y|z) on the whitened CLIP concept scores instead (label-free, no "
        "groups). Adds a '_grw_<mode>' suffix.",
    )
    parser.add_argument(
        "--clip_model",
        default=DebiasingConfig.clip_model,
        help="Frozen image-text encoder: a Hugging Face CLIP/SigLIP id (e.g. "
        "google/siglip-base-patch16-224) or an OpenCLIP hf-hub:<repo> id (e.g. "
        "hf-hub:timm/PE-Core-B-16).  A non-default encoder adds an "
        "'_enc-<name>' suffix to every run.",
    )
    parser.add_argument(
        "--extra_prompts",
        default=None,
        help="JSON file from discover_biases.py: its accepted (pos, neg) prompt "
        "pairs are added as further concepts (subspace, eraser, pseudo-groups). "
        "Adds an '_xp-<file stem>' suffix.",
    )
    parser.add_argument(
        "--core_dirs",
        default=None,
        help="JSON file from discover_core.py: its accepted core directions are "
        "promoted (a head on E(x) predicts their whitened scores, residual to the "
        "concept subspace).  Needs --lambda_promote > 0.",
    )
    parser.add_argument(
        "--lambda_promote",
        type=float,
        default=0.0,
        help="Weight of the core-promotion loss for runs with a debiasing term. "
        "Adds a '_core_lpro<v>' suffix.",
    )
    parser.add_argument(
        "--prompts",
        default="words",
        choices=list(PROMPT_SETS) + ["none"],
        help="Concept prompt set for the subspace, eraser and pseudo-labels "
        "('none': only the --extra_prompts pairs). "
        "Anything but 'words' adds a '_p<set>' suffix to every run.",
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
        clip_model=args.clip_model,
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
    # CLIP is always loaded: val pseudo-labels drive best_pseudo_wga.pt for every run
    print(f"\nLoading CLIP ({ref_cfg.clip_model}) …")
    oracle = CLIPOracle(ref_cfg.clip_model, device=device)
    print(f"  embed dim {oracle.embed_dim}, input {oracle.image_size}px")

    # pos_words = get_synonyms("water")
    # neg_words = get_random_base_nouns(n=len(pos_words)*5, exclude_words=set(pos_words), seed=42)
    prompt_pairs = [] if args.prompts == "none" else [PROMPT_SETS[args.prompts]]
    print(f"  prompts: {args.prompts}")
    if args.extra_prompts:
        prompt_pairs = prompt_pairs + load_prompt_file(args.extra_prompts)
        print(f"  + confirmed prompt pairs from {args.extra_prompts}")
    if not prompt_pairs:
        raise SystemExit("--prompts none needs --extra_prompts")
    for pos, neg in prompt_pairs:
        print(f"  pos: {pos}")
        print(f"  neg: {neg}")
    prompts_pos, prompts_neg = prompt_pairs[0]
    custom_pairs = args.prompts == "none" or bool(args.extra_prompts)

    print(f"Computing concept subspace (k={args.concept_subspace_k}) …")
    subspace = compute_multi_concept_subspace(
        oracle, prompt_pairs, k=args.concept_subspace_k
    )
    print(f"  subspace shape: {tuple(subspace.shape)}")

    # ── Val pseudo-labels: label-free checkpoint selection (best_pseudo_wga.pt) ─
    print("\nEmbedding the val split with CLIP …")
    v_va, _, c_va = embed_dataset(
        oracle, val_loader.dataset, num_workers=ref_cfg.num_workers
    )

    # ── One CLIP pass over train: whitened eraser and/or group labels ─────────
    if bool(args.core_dirs) != (args.lambda_promote > 0):
        raise SystemExit("--core_dirs and --lambda_promote > 0 go together")
    need_eraser = (
        args.eraser == "leace"
        or args.align == "cka"
        or args.repulse != "orth"
        or args.group_reweight == "propensity"
        or args.lambda_promote > 0
    )
    eraser = core_eraser = train_weights = cutoffs = None
    if need_eraser or args.group_reweight != "none" or args.zs_cutoff != "zero":
        print("\nEmbedding the train split with CLIP …")
        v_tr, y_tr, c_tr = embed_dataset(
            oracle, train_loader.dataset, num_workers=ref_cfg.num_workers
        )
        if args.zs_cutoff == "gmm":
            cutoffs = fit_zeroshot_cutoffs(oracle, prompt_pairs, v_tr)
            print(f"  fitted zero-shot cutoffs (unlabeled train): {cutoffs}")
        if need_eraser:
            fit_w = None
            if args.eraser_fit == "balanced":
                c_fit = zeroshot_concept_codes(oracle, prompt_pairs, v_tr, cutoffs)
                fit_w = group_weights(y_tr, c_fit)
            eraser = fit_leace_eraser(v_tr, subspace, weights=fit_w)
            if args.core_dirs:
                core = load_core_dirs(args.core_dirs)
                core_eraser = fit_leace_eraser(
                    v_tr, torch.cat([subspace.float().cpu(), core]), weights=fit_w
                )
                print(f"  {len(core)} core direction(s) from {args.core_dirs}")
            print(
                f"  fitted whitened eraser on {len(v_tr):,} train embeddings "
                f"({args.eraser_fit})"
            )
        if args.group_reweight == "pseudo":
            c_hat = zeroshot_concept_codes(oracle, prompt_pairs, v_tr, cutoffs)
            print(
                "  zero-shot concept accuracy on train (reported only, not used): "
                f"{(c_hat == c_tr).float().mean():.4f}"
            )
            train_weights = group_weights(y_tr, c_hat)
        elif args.group_reweight == "oracle":
            train_weights = group_weights(y_tr, c_tr)
        elif args.group_reweight == "propensity":
            z_tr = whitened_concept_scores(v_tr, eraser)
            train_weights = propensity_weights(y_tr, z_tr)

    val_pseudo = zeroshot_concept_codes(oracle, prompt_pairs, v_va, cutoffs)
    print(
        "  zero-shot concept accuracy on val (reported only, not used): "
        f"{(val_pseudo == c_va).float().mean():.4f}"
    )

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
                sequential_phase_switch_epoch=args.sequential_switch_epoch,
                clip_model=args.clip_model,
                clip_embed_dim=oracle.embed_dim,
                proj_out_dim=oracle.embed_dim,
            )
            cfg.concept_attr = "place"
            cfg.concept_prompts_pos, cfg.concept_prompts_neg = prompts_pos, prompts_neg
            if custom_pairs:
                cfg.prompt_pairs = [[list(p), list(n)] for p, n in prompt_pairs]
            if args.extra_prompts:
                cfg.run_name += "_xp-" + Path(args.extra_prompts).stem
            if args.prompts != "words":
                cfg.run_name += f"_p{args.prompts}"
            if args.clip_model != DebiasingConfig.clip_model:
                cfg.run_name += "_enc-" + args.clip_model.split("/")[-1]
            if args.no_lambda_schedules:
                cfg = disable_lambda_schedules(cfg)
            if args.scratch:
                cfg = use_scratch_backbone(cfg)
            if args.eraser == "leace":
                cfg = use_leace_eraser(cfg)
            if args.repulse != "orth":
                cfg = use_whitened_repulse(cfg, args.repulse)
            if args.align == "cka" and cfg.lambda_align > 0:
                cfg.align = "cka"
                cfg.run_name += "_acka"
            if args.lambda_align is not None and cfg.lambda_align > 0:
                cfg.lambda_align = args.lambda_align
                cfg.run_name += f"_lal{args.lambda_align:g}"
            if args.eraser_fit != "biased":
                cfg.eraser_fit = args.eraser_fit
                cfg.run_name += "_efbal"
            if args.zs_cutoff != "zero":
                cfg.zs_cutoff, cfg.zs_cutoffs = args.zs_cutoff, cutoffs
                cfg.run_name += f"_zs{args.zs_cutoff}"
            if args.lambda_repulse is not None and cfg.lambda_repulse > 0:
                cfg.lambda_repulse = args.lambda_repulse
                cfg.run_name += f"_lrep{args.lambda_repulse:g}"
            if args.lambda_promote > 0 and run_name != "erm":
                cfg.core_dirs, cfg.lambda_promote = args.core_dirs, args.lambda_promote
                cfg.run_name += f"_core_lpro{args.lambda_promote:g}"
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
                core_eraser=None if is_erm else core_eraser,
            )
            fit_loader = (
                weighted_loader(train_loader, train_weights, seed=seed)
                if train_weights is not None
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
