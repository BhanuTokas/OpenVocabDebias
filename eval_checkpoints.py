#!/usr/bin/env python
"""
eval_checkpoints.py — Re-evaluate saved runs without re-training.

For every run directory found under the given paths, evaluates each selected
checkpoint (default: best.pt and best_wga.pt) on the test split and appends one
row per (run, checkpoint) to a summary CSV.  Each row reports the concept probe
on both E(x) and P(E(x)), task accuracy and worst-group accuracy.

    best.pt            — selected by val accuracy      (label-free model selection)
    best_wga.pt        — selected by val WGA           (needs concept labels on val)
    best_pseudo_wga.pt — selected by pseudo-group val WGA, with CLIP zero-shot
                         concept labels on val         (label-free)

Runs trained before best_pseudo_wga.pt existed can get one after the fact with
--pseudo_from_epochs: the epoch_XX.pt checkpoints are scored on val with CLIP
pseudo-groups and the best one is written to the run dir (with a .json noting
the chosen epoch).  Existing best_pseudo_wga.pt files are never overwritten.

Run directories are folders containing at least one of the checkpoints.  New
runs (<run_name>/seed_N/<timestamp>/) carry a config.json, which is used to
rebuild the model and data.  Legacy runs (<run_name>/seed_N/, no config.json)
fall back to the CLI defaults below; note those were overwritten by every
re-run of the same run_name/seed, so they only reflect the last sweep.

Usage
-----
    # All runs of a sweep
    python eval_checkpoints.py checkpoints_waterbirds/sched_control \\
        --out results_waterbirds/sched_control/eval_summary.csv

    # Add label-free selection to runs that predate best_pseudo_wga.pt
    python eval_checkpoints.py checkpoints_waterbirds/leace_pgrw --pseudo_from_epochs \\
        --out results_waterbirds/leace_pgrw/eval_summary_pseudo.csv

    # Specific run dirs, CelebA path differs from the one in config.json
    python eval_checkpoints.py checkpoints/full/seed_42/20260710-115918 \\
        --celeba_root /data/hkerner/Datasets/CelebA/
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import fields
from typing import Dict, List, Tuple

import numpy as np
import torch

from clip_debias.clip_oracle import CLIPOracle, embed_dataset, zeroshot_concept_codes
from clip_debias.config import DebiasingConfig, get_concept_prompts
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import extract_features, run_evaluation, worst_group_accuracy
from clip_debias.models import build_model
from clip_debias.results_io import append_result
from clip_debias.waterbirds_data import build_waterbirds_dataloaders

SELECTION = {
    "best.pt": "val_acc",
    "best_wga.pt": "val_wga",
    "best_pseudo_wga.pt": "val_pseudo_wga",
}
PSEUDO_CKPT = "best_pseudo_wga.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints (E(x) and P(E(x)) probes)"
    )
    parser.add_argument(
        "paths", nargs="+", help="Run dirs, or parent dirs to search recursively"
    )
    parser.add_argument("--ckpts", nargs="+", default=list(SELECTION))
    parser.add_argument(
        "--pseudo_from_epochs",
        action="store_true",
        help=f"Create {PSEUDO_CKPT} from epoch_XX.pt for runs that lack it",
    )
    parser.add_argument("--out", default="./results_eval/eval_summary.csv")
    parser.add_argument(
        "--celeba_root",
        default=None,
        help="Override the CelebA root stored in config.json (paths differ per machine)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")

    legacy = parser.add_argument_group(
        "legacy runs", "Used only for run dirs without a config.json"
    )
    legacy.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "celeba", "waterbirds"],
        help="auto: waterbirds if 'waterbirds' is in the path, else celeba",
    )
    legacy.add_argument(
        "--backbone", default="resnet18", choices=["resnet18", "resnet50", "vit_b_16"]
    )
    legacy.add_argument("--target_attr", default="Attractive")
    legacy.add_argument("--concept_attr", default="Male")
    return parser.parse_args()


def find_run_dirs(
    paths: List[str], ckpt_names: List[str], with_epochs: bool = False
) -> List[str]:
    """Dirs holding any of ckpt_names (or, with_epochs, any epoch_XX.pt — so runs
    that best_pseudo_wga.pt is still to be created for are found too)."""
    run_dirs = set()
    for path in paths:
        for root, _dirs, files in os.walk(path):
            if any(name in files for name in ckpt_names) or (
                with_epochs and any(re.fullmatch(r"epoch_\d+\.pt", f) for f in files)
            ):
                run_dirs.add(os.path.normpath(root))
    return sorted(run_dirs)


def load_run_cfg(run_dir: str, args) -> Tuple[DebiasingConfig, str, str]:
    """Return (cfg, dataset, config_source) for a run directory."""
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            raw = json.load(f)["config"]
        known = {f.name for f in fields(DebiasingConfig)}
        cfg = DebiasingConfig(**{k: v for k, v in raw.items() if k in known})
        dataset = "waterbirds" if cfg.concept_attr == "place" else "celeba"
        source = "config.json"
    else:
        dataset = args.dataset
        if dataset == "auto":
            dataset = "waterbirds" if "waterbirds" in run_dir.lower() else "celeba"
        cfg = DebiasingConfig(
            backbone=args.backbone,
            target_attr=args.target_attr,
            concept_attr="place" if dataset == "waterbirds" else args.concept_attr,
        )
        # <run_name>/seed_N  →  run_name, seed
        parts = os.path.normpath(run_dir).split(os.sep)
        for i, part in enumerate(parts):
            m = re.fullmatch(r"seed_(\d+)", part)
            if m and i > 0:
                cfg.seed = int(m.group(1))
                cfg.run_name = parts[i - 1]
        source = "legacy"

    if args.celeba_root is not None:
        cfg.celeba_root = args.celeba_root
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.amp = not args.no_amp
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    return cfg, dataset, source


def concept_prompts(cfg: DebiasingConfig, dataset: str) -> Tuple[List[str], List[str]]:
    """Prompts the run used for its concept.  Waterbirds configs written before
    Oct 2026 recorded the CelebA gender defaults instead; those runs used the
    Waterbirds word prompts."""
    if dataset == "waterbirds":
        from waterbirds_train import PROMPT_SETS

        male_pos, _ = get_concept_prompts("Male")
        if not cfg.concept_prompts_pos or list(cfg.concept_prompts_pos) == list(
            male_pos
        ):
            return PROMPT_SETS["words"]
        return cfg.concept_prompts_pos, cfg.concept_prompts_neg
    if cfg.concept_prompts_pos and cfg.concept_prompts_neg:
        return cfg.concept_prompts_pos, cfg.concept_prompts_neg
    return get_concept_prompts(cfg.concept_attr)


def select_pseudo_best(
    run_dir: str, model, val_loader, val_pseudo: np.ndarray, cfg
) -> bool:
    """Write run_dir/best_pseudo_wga.pt from the epoch checkpoint with the best
    pseudo-group val WGA (first one on ties, as in the trainer)."""
    epoch_ckpts = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pt")))
    if not epoch_ckpts:
        print(f"  [skip] no epoch_*.pt in {run_dir}; cannot create {PSEUDO_CKPT}")
        return False
    best = None
    for path in epoch_ckpts:
        state = torch.load(path, map_location=cfg.device)
        model.load_state_dict(state["model_state"])
        feats = extract_features(model, val_loader, cfg.device, use_amp=cfg.amp)
        _, wga = worst_group_accuracy(feats["targets"], val_pseudo, feats["preds"])
        print(f"    {os.path.basename(path)}  val_pseudo_wga={wga:.4f}")
        if best is None or wga > best[0]:
            best = (wga, state["epoch"], state["model_state"])
    torch.save(best[2], os.path.join(run_dir, PSEUDO_CKPT))
    with open(os.path.join(run_dir, "best_pseudo_wga.json"), "w") as f:
        json.dump(
            {
                "epoch": best[1],
                "val_pseudo_wga": best[0],
                "created_by": "eval_checkpoints.py",
            },
            f,
            indent=2,
        )
    print(f"  ✓ {PSEUDO_CKPT} ← epoch {best[1]}  (val_pseudo_wga={best[0]:.4f})")
    return True


def print_summary(rows: List[dict]):
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["run_name"], r["checkpoint"])].append(r)

    metric_keys = [
        "probe_test_acc",
        "probe_proj_test_acc",
        "task_test_acc",
        "worst_group_acc",
    ]
    print(f"\n{'='*100}")
    print("  Summary (mean ± std across seeds)   probe = E(x), probe_proj = P(E(x))")
    print(f"{'='*100}")
    run_w = max([22] + [len(r) + 2 for r, _ in grouped])
    ckpt_w = max([13] + [len(c) + 2 for _, c in grouped])
    print(
        f"  {'run':<{run_w}}{'ckpt':<{ckpt_w}}"
        + "".join(f"{k:<22}" for k in metric_keys)
    )
    for (run_name, ckpt), group in sorted(grouped.items()):
        line = f"  {run_name:<{run_w}}{ckpt:<{ckpt_w}}"
        for k in metric_keys:
            vals = [g[k] for g in group]
            line += f"{np.mean(vals):.4f} ± {np.std(vals):.4f}     "
        print(line + f"(n={len(group)})")
    print()


def main():
    args = parse_args()
    run_dirs = find_run_dirs(
        args.paths, args.ckpts, with_epochs=args.pseudo_from_epochs
    )
    if not run_dirs:
        raise SystemExit(f"No run dirs containing {args.ckpts} under {args.paths}")
    print(f"Found {len(run_dirs)} run dir(s)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader_cache: Dict[tuple, tuple] = {}
    pseudo_cache: Dict[tuple, np.ndarray] = {}
    oracle, oracle_name = None, None
    all_rows: List[dict] = []

    for run_dir in run_dirs:
        cfg, dataset, source = load_run_cfg(run_dir, args)

        key = (
            dataset,
            cfg.celeba_root,
            cfg.target_attr,
            cfg.concept_attr,
            tuple(cfg.extra_concept_attrs),
        )
        if key not in loader_cache:
            print(f"\nBuilding {dataset} dataloaders …")
            if dataset == "waterbirds":
                loader_cache[key] = build_waterbirds_dataloaders(cfg)
            else:
                loader_cache[key] = build_dataloaders(cfg)
        train_loader, val_loader, test_loader = loader_cache[key]

        model = build_model(cfg, num_classes=2)
        if args.pseudo_from_epochs and not os.path.isfile(
            os.path.join(run_dir, PSEUDO_CKPT)
        ):
            pairs = [concept_prompts(cfg, dataset)] + [
                get_concept_prompts(a) for a in cfg.extra_concept_attrs
            ]
            pkey = key + (
                cfg.clip_model,
                tuple(tuple(p) + tuple(n) for p, n in pairs),
                tuple(cfg.zs_cutoffs),
            )
            if pkey not in pseudo_cache:
                if oracle is None or oracle_name != cfg.clip_model:
                    print(f"\nLoading CLIP ({cfg.clip_model}) …")
                    oracle, oracle_name = (
                        CLIPOracle(cfg.clip_model, device=cfg.device),
                        cfg.clip_model,
                    )
                v_va, _, c_va = embed_dataset(
                    oracle, val_loader.dataset, num_workers=args.num_workers
                )
                c_hat = zeroshot_concept_codes(oracle, pairs, v_va, cfg.zs_cutoffs)
                print(
                    f"  zero-shot concept accuracy on val: {(c_hat == c_va).float().mean():.4f}"
                )
                pseudo_cache[pkey] = c_hat.numpy()
            print(
                f"\n[{cfg.run_name}  seed={cfg.seed}] selecting {PSEUDO_CKPT} from epoch checkpoints"
            )
            select_pseudo_best(run_dir, model, val_loader, pseudo_cache[pkey], cfg)
        for ckpt_name in args.ckpts:
            ckpt_path = os.path.join(run_dir, ckpt_name)
            if not os.path.isfile(ckpt_path):
                print(f"  [skip] {ckpt_path} not found")
                continue

            state = torch.load(ckpt_path, map_location=cfg.device)
            if "model_state" in state:
                state = state["model_state"]
            model.load_state_dict(state)

            torch.manual_seed(0)
            metrics = run_evaluation(
                model,
                train_loader,
                test_loader,
                cfg.device,
                cfg.amp,
                label=f"{cfg.run_name}  seed={cfg.seed}  {ckpt_name}",
            )

            row = {
                "run_name": cfg.run_name,
                "seed": cfg.seed,
                "dataset": dataset,
                "target_attr": "label" if dataset == "waterbirds" else cfg.target_attr,
                "concept_attr": cfg.concept_attr,
                "subspace_k": cfg.concept_subspace_k if source == "config.json" else "",
                "checkpoint": ckpt_name,
                "selected_by": SELECTION.get(ckpt_name, ""),
                "config_source": source,
                "run_dir": run_dir,
                **metrics,
            }
            all_rows.append(row)
            append_result(args.out, row)

    print_summary(all_rows)
    print(f"Per-checkpoint results saved to: {args.out}")


if __name__ == "__main__":
    main()
