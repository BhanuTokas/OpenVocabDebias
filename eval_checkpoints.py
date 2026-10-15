#!/usr/bin/env python
"""
eval_checkpoints.py — Re-evaluate saved runs without re-training.

For every run directory found under the given paths, evaluates each selected
checkpoint (default: best.pt and best_wga.pt) on the test split and appends one
row per (run, checkpoint) to a summary CSV.  Each row reports the concept probe
on both E(x) and P(E(x)), task accuracy and worst-group accuracy.

    best.pt     — selected by val accuracy  (label-free model selection)
    best_wga.pt — selected by val WGA       (needs concept labels on val)

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

    # Specific run dirs, CelebA path differs from the one in config.json
    python eval_checkpoints.py checkpoints/full/seed_42/20260710-115918 \\
        --celeba_root /data/hkerner/Datasets/CelebA/
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import fields
from typing import Dict, List, Tuple

import numpy as np
import torch

from clip_debias.config import DebiasingConfig
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.results_io import append_result
from clip_debias.waterbirds_data import build_waterbirds_dataloaders

SELECTION = {
    "best.pt": "val_acc",
    "best_wga.pt": "val_wga",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints (E(x) and P(E(x)) probes)"
    )
    parser.add_argument(
        "paths", nargs="+", help="Run dirs, or parent dirs to search recursively"
    )
    parser.add_argument("--ckpts", nargs="+", default=list(SELECTION))
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


def find_run_dirs(paths: List[str], ckpt_names: List[str]) -> List[str]:
    run_dirs = set()
    for path in paths:
        for root, _dirs, files in os.walk(path):
            if any(name in files for name in ckpt_names):
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
    print(f"  {'run':<22}{'ckpt':<13}" + "".join(f"{k:<22}" for k in metric_keys))
    for (run_name, ckpt), group in sorted(grouped.items()):
        line = f"  {run_name:<22}{ckpt:<13}"
        for k in metric_keys:
            vals = [g[k] for g in group]
            line += f"{np.mean(vals):.4f} ± {np.std(vals):.4f}     "
        print(line + f"(n={len(group)})")
    print()


def main():
    args = parse_args()
    run_dirs = find_run_dirs(args.paths, args.ckpts)
    if not run_dirs:
        raise SystemExit(f"No run dirs containing {args.ckpts} under {args.paths}")
    print(f"Found {len(run_dirs)} run dir(s)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader_cache: Dict[tuple, tuple] = {}
    all_rows: List[dict] = []

    for run_dir in run_dirs:
        cfg, dataset, source = load_run_cfg(run_dir, args)

        key = (dataset, cfg.celeba_root, cfg.target_attr, cfg.concept_attr)
        if key not in loader_cache:
            print(f"\nBuilding {dataset} dataloaders …")
            if dataset == "waterbirds":
                loader_cache[key] = build_waterbirds_dataloaders(cfg)
            else:
                loader_cache[key] = build_dataloaders(cfg)
        train_loader, _val_loader, test_loader = loader_cache[key]

        model = build_model(cfg, num_classes=2)
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
