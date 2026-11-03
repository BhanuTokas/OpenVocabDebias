#!/usr/bin/env python
"""
discover_biases.py — Propose prompts for biases a trained model still relies on.

For one run directory (with config.json) and checkpoint, the model's mistakes
on the val split give within-class error directions in CLIP space; the concepts
the run was already prompted with are removed (joint whitened fit) and each
remaining direction is named by vocabulary phrases (clip_debias/discovery.py).
The proposals are written to an editable JSON file:

    {"proposals": [{"name": ..., "pos": [...], "neg": [...], "accept": false,
                    "class": k, "scores": {...}}, ...]}

Review it, delete wrong phrases (e.g. words for the target class itself), set
"accept": true, and pass the file to training with --extra_prompts.  Only
target labels are used.  The errors come from (--split):
  val    the val split, which the model has not trained on (default)
  train  the train split without augmentation, so val is left for model
         selection only.  A converged model has nearly memorised train, so use
         an early checkpoint (--ckpt epoch_01.pt), or cross-fit: pass the K run
         directories of one K-fold split (training flags --n_folds K --fold k
         --fold_seed s) and each train example is predicted by the run that
         held it out.
The unit directions are also saved next to --out (<out stem>_dirs.npz) so the
discoveries of different splits / seeds can be compared.

With --report_truth, the correlation of each discovered score with the
dataset's true concept attribute(s) is printed (analysis only).

Which end of a discovered pair is "pos" is arbitrary, so the "zero-shot concept
accuracy" that training logs for its true concept can come out inverted (e.g.
0.10 instead of 0.90); the pseudo-groups are the same either way.

Usage
-----
    python discover_biases.py checkpoints/.../seed_42/<timestamp> \\
        --celeba_root /path/to/CelebA --out proposals.json --report_truth
    python discover_biases.py <waterbirds ERM run dir> --prompted none --out wb.json
    python discover_biases.py <ERM run dir> --split train --ckpt epoch_01.pt ...
    python discover_biases.py <fold 0 dir> ... <fold K-1 dir> --split train \\
        --ckpt epoch_01.pt --prompted none --out wb_cf.json
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clip_debias.baselines import fold_indices
from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_multi_concept_subspace,
    embed_dataset,
)
from clip_debias.config import get_concept_prompts
from clip_debias.data import build_dataloaders
from clip_debias.discovery import (
    as_prompt,
    error_directions,
    name_scores,
    residual_scores,
    vocabulary,
)
from clip_debias.evaluate import extract_features
from clip_debias.models import build_model
from clip_debias.waterbirds_data import build_waterbirds_dataloaders
from error_set import train_eval_dataset
from eval_checkpoints import concept_prompts, load_run_cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "run_dir",
        nargs="+",
        help="run directory containing config.json and the checkpoint, or the K "
        "fold runs of one cross-fitting split (with --split train)",
    )
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--ckpt", default="best_pseudo_wga.pt")
    p.add_argument(
        "--prompted",
        default="config",
        choices=["config", "none"],
        help="Remove the concepts the run was prompted with (config) or nothing",
    )
    p.add_argument("--celeba_root", default=None)
    p.add_argument(
        "--objects", action="store_true", help="add ImageNet classes to the vocabulary"
    )
    p.add_argument("--n_phrases", type=int, default=5)
    p.add_argument("--out", default="proposals.json")
    p.add_argument("--report_truth", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def prompted_pairs(cfg, dataset):
    if cfg.prompt_pairs:
        return [(list(p), list(n)) for p, n in cfg.prompt_pairs]
    return [concept_prompts(cfg, dataset)] + [
        get_concept_prompts(a) for a in cfg.extra_concept_attrs
    ]


def model_errors(run_dir, cfg, ds, idx, args):
    """(targets, error mask) of the run's --ckpt on ds[idx], in idx order."""
    model = build_model(cfg, num_classes=2)
    state = torch.load(os.path.join(run_dir, args.ckpt), map_location=cfg.device)
    model.load_state_dict(state.get("model_state", state))
    loader = DataLoader(
        Subset(ds, idx.tolist()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    feats = extract_features(model, loader, cfg.device, use_amp=cfg.amp)
    return feats["targets"].astype(int), feats["preds"] != feats["targets"]


def main():
    args = parse_args()
    runs = [
        load_run_cfg(d, SimpleNamespace(**vars(args), dataset="auto"))[:2]
        for d in args.run_dir
    ]
    cfg, dataset = runs[0]
    if args.split == "val":
        loaders = (
            build_waterbirds_dataloaders(cfg)
            if dataset == "waterbirds"
            else build_dataloaders(cfg)
        )
        ds = loaders[1].dataset
    else:
        ds = train_eval_dataset(cfg, dataset)
    n = len(ds)

    if len(runs) > 1 or cfg.n_folds > 0:
        K, s = cfg.n_folds, cfg.fold_seed
        if (
            args.split != "train"
            or K == 0
            or any((c.n_folds, c.fold_seed) != (K, s) for c, _ in runs)
            or sorted(c.fold for c, _ in runs) != list(range(K))
        ):
            raise SystemExit(
                "cross-fitting needs --split train and one run per fold 0..K-1 of "
                "the same split (--n_folds K --fold k --fold_seed s)"
            )
        y, err = np.zeros(n, int), np.zeros(n, bool)
        for d, (c, _) in zip(args.run_dir, runs):
            idx = fold_indices(n, K, c.fold, s)
            y[idx], err[idx] = model_errors(d, c, ds, idx, args)
        source = f"held-out train ({K} folds, split seed {s})"
    else:
        y, err = model_errors(args.run_dir[0], cfg, ds, np.arange(n), args)
        source = args.split
    print(
        f"\n{cfg.run_name}  seed={cfg.seed}  {args.ckpt}: {source} error {err.mean():.3f}"
    )
    for k in np.unique(y):
        print(
            f"  class {k}: {int((err & (y == k)).sum())} errors / {int((y == k).sum())}"
        )

    oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
    v, _, c_true = embed_dataset(oracle, ds, num_workers=args.num_workers)
    pairs = prompted_pairs(cfg, dataset) if args.prompted == "config" else []
    S = (
        compute_multi_concept_subspace(oracle, pairs, k=cfg.concept_subspace_k)
        if pairs
        else None
    )
    if pairs:
        print(f"  removing {len(pairs)} prompted concept(s), subspace {tuple(S.shape)}")

    dirs, classes = error_directions(v, y, err)
    z = residual_scores(v, S, dirs)
    vocab = vocabulary(objects=args.objects)
    proposals = []
    for j, k in enumerate(classes):
        pos, neg = name_scores(oracle, vocab, v, z[:, j], n=args.n_phrases)
        entry = dict(
            name=f"discovered_class{k}",
            pos=[as_prompt(p) for p, _ in pos],
            neg=[as_prompt(p) for p, _ in neg],
            accept=False,
            **{"class": k},
            scores=dict(
                pos={p: round(s, 3) for p, s in pos},
                neg={p: round(s, 3) for p, s in neg},
                corr_y=round(float(np.corrcoef(z[:, j], y)[0, 1]), 3),
            ),
        )
        print(
            f"\n  direction from class {k} errors   corr(score, y) {entry['scores']['corr_y']:+.2f}"
        )
        print("    + " + ", ".join(f"{p} ({s:+.2f})" for p, s in pos))
        print("    − " + ", ".join(f"{p} ({s:+.2f})" for p, s in neg))
        if args.report_truth:
            n_attr = 1 + len(cfg.extra_concept_attrs)
            names = [cfg.concept_attr] + list(cfg.extra_concept_attrs)
            truth = {
                names[b]: round(
                    float(np.corrcoef(z[:, j], (c_true.numpy() >> b) & 1)[0, 1]), 3
                )
                for b in range(n_attr)
            }
            entry["truth_corr"] = truth
            print(
                "    (analysis) corr with true "
                + "  ".join(f"{a} {t:+.2f}" for a, t in truth.items())
            )
        proposals.append(entry)

    out = dict(
        run_dir=args.run_dir if len(args.run_dir) > 1 else args.run_dir[0],
        split=args.split,
        ckpt=args.ckpt,
        clip_model=cfg.clip_model,
        prompted=[list(p) for p in pairs],
        proposals=proposals,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(
        os.path.splitext(args.out)[0] + "_dirs.npz",
        dirs=dirs.numpy(),
        classes=np.array(classes),
        n_errors=np.array([int((err & (y == k)).sum()) for k in classes]),
    )
    print(
        f'\nProposals written to {args.out}: review, edit, set "accept": true, '
        "then train with --extra_prompts"
    )


if __name__ == "__main__":
    main()
