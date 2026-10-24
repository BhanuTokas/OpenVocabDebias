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
"accept": true, and pass the file to training with --extra_prompts.  The val
split is used because the model has not trained on it (train errors are rare
after memorisation and come with augmentation); only target labels are used.

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
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from clip_debias.clip_oracle import CLIPOracle, compute_multi_concept_subspace, embed_dataset
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
from eval_checkpoints import concept_prompts, load_run_cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="run directory containing config.json and the checkpoint")
    p.add_argument("--ckpt", default="best_pseudo_wga.pt")
    p.add_argument(
        "--prompted",
        default="config",
        choices=["config", "none"],
        help="Remove the concepts the run was prompted with (config) or nothing",
    )
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--objects", action="store_true", help="add ImageNet classes to the vocabulary")
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


def main():
    args = parse_args()
    cfg, dataset, _ = load_run_cfg(
        args.run_dir, SimpleNamespace(**vars(args), dataset="auto")
    )
    loaders = (
        build_waterbirds_dataloaders(cfg) if dataset == "waterbirds" else build_dataloaders(cfg)
    )
    val_ds = loaders[1].dataset

    model = build_model(cfg, num_classes=2)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=cfg.device)
    model.load_state_dict(state.get("model_state", state))
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    feats = extract_features(model, loader, cfg.device, use_amp=cfg.amp)
    y, err = feats["targets"].astype(int), feats["preds"] != feats["targets"]
    print(f"\n{cfg.run_name}  seed={cfg.seed}  {args.ckpt}: val error {err.mean():.3f}")
    for k in np.unique(y):
        print(f"  class {k}: {int((err & (y == k)).sum())} errors / {int((y == k).sum())}")

    oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
    v, _, c_true = embed_dataset(oracle, val_ds, num_workers=args.num_workers)
    pairs = prompted_pairs(cfg, dataset) if args.prompted == "config" else []
    S = compute_multi_concept_subspace(oracle, pairs, k=cfg.concept_subspace_k) if pairs else None
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
        print(f"\n  direction from class {k} errors   corr(score, y) {entry['scores']['corr_y']:+.2f}")
        print("    + " + ", ".join(f"{p} ({s:+.2f})" for p, s in pos))
        print("    − " + ", ".join(f"{p} ({s:+.2f})" for p, s in neg))
        if args.report_truth:
            n_attr = 1 + len(cfg.extra_concept_attrs)
            names = [cfg.concept_attr] + list(cfg.extra_concept_attrs)
            truth = {
                names[b]: round(float(np.corrcoef(z[:, j], (c_true.numpy() >> b) & 1)[0, 1]), 3)
                for b in range(n_attr)
            }
            entry["truth_corr"] = truth
            print("    (analysis) corr with true " + "  ".join(f"{a} {t:+.2f}" for a, t in truth.items()))
        proposals.append(entry)

    out = dict(
        run_dir=args.run_dir,
        ckpt=args.ckpt,
        clip_model=cfg.clip_model,
        prompted=[list(p) for p in pairs],
        proposals=proposals,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nProposals written to {args.out}: review, edit, set \"accept\": true, "
          "then train with --extra_prompts")


if __name__ == "__main__":
    main()
