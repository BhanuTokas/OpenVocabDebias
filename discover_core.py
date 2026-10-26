#!/usr/bin/env python
"""
discover_core.py — Find core concepts a trained model still under-uses.

The mirror image of discover_biases.py.  On the val split, the run's prompted
concepts give CLIP zero-shot pseudo-groups; within each (class × pseudo-group)
cell the bias is held fixed, so mean(correct) − mean(errors) of the CLIP image
embeddings points at what the model is missing (clip_debias/discovery.py,
core_directions).  The pooled direction and the per-cell ones are written to a
JSON file, each with phrases that name it (explanation only: core directions
are used as image-space vectors, because part words carry the bias in CLIP
text space):

    {"directions": [{"tag": "pooled", "accept": true, "vector": [...],
                     "names": {"pos": [...], "neg": [...]}}, ...]}

The pooled direction is accepted by default; training uses the accepted ones
with --core_dirs (and --lambda_promote).  Only target labels are used; with
--report_truth, each direction's score is also checked against the dataset's
true concept attribute(s) (analysis only).

Usage
-----
    python discover_core.py <run dir> --out core.json --vocab birds --report_truth
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_multi_concept_subspace,
    embed_dataset,
    fit_zeroshot_cutoff,
    zeroshot_concept_codes,
)
from clip_debias.data import build_dataloaders
from clip_debias.discovery import (
    CORE_VOCAB,
    core_directions,
    name_scores,
    residual_scores,
    vocabulary,
)
from clip_debias.evaluate import extract_features
from clip_debias.models import build_model
from clip_debias.waterbirds_data import build_waterbirds_dataloaders
from discover_biases import prompted_pairs
from eval_checkpoints import load_run_cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="run directory containing config.json and the checkpoint")
    p.add_argument("--ckpt", default="best_pseudo_wga.pt")
    p.add_argument("--celeba_root", default=None)
    p.add_argument(
        "--vocab",
        default="birds",
        choices=list(CORE_VOCAB) + ["generic"],
        help="phrases used to name the directions (explanation only)",
    )
    p.add_argument("--min_errors", type=int, default=20)
    p.add_argument("--n_phrases", type=int, default=5)
    p.add_argument("--out", default="core.json")
    p.add_argument("--report_truth", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


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

    oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
    v, _, c_true = embed_dataset(oracle, val_ds, num_workers=args.num_workers)
    pairs = prompted_pairs(cfg, dataset)
    S = compute_multi_concept_subspace(oracle, pairs, k=cfg.concept_subspace_k)
    groups = zeroshot_concept_codes(oracle, pairs, v, cfg.zs_cutoffs or None).numpy()
    print(f"  {len(pairs)} prompted concept(s) → pseudo-groups; subspace {tuple(S.shape)}")

    dirs, tags = core_directions(v, y, groups, err, min_errors=args.min_errors)
    z = residual_scores(v, S, dirs)
    vocab = vocabulary() if args.vocab == "generic" else CORE_VOCAB[args.vocab]
    n_attr = 1 + len(cfg.extra_concept_attrs)
    names = [cfg.concept_attr] + list(cfg.extra_concept_attrs)
    entries = []
    for j, tag in enumerate(tags):
        zj = z[:, j] * float(np.sign(np.corrcoef(z[:, j], y)[0, 1]) or 1.0)  # toward class 1
        pos, neg = name_scores(oracle, vocab, v, zj, n=args.n_phrases)
        entry = dict(
            tag=tag,
            accept=tag == "pooled",
            names=dict(pos=[p for p, _ in pos], neg=[p for p, _ in neg]),
            corr_y=round(float(np.corrcoef(zj, y)[0, 1]), 3),
        )
        print(f"\n  [{tag}]  corr(score, y) {entry['corr_y']:+.2f}")
        print("    + " + ", ".join(f"{p} ({s:+.2f})" for p, s in pos))
        print("    − " + ", ".join(f"{p} ({s:+.2f})" for p, s in neg))
        if args.report_truth:
            zz = zj.numpy() - fit_zeroshot_cutoff(zj)
            truth = {}
            for b in range(n_attr):
                cb = (c_true.numpy() >> b) & 1
                acc = {
                    f"y{a}_{names[b]}{t}": round(float(((zz > 0) == a)[(y == a) & (cb == t)].mean()), 3)
                    for a in (0, 1)
                    for t in (0, 1)
                }
                truth[names[b]] = dict(corr=round(float(np.corrcoef(zz, cb)[0, 1]), 3), y_acc=acc)
                print(
                    f"    (analysis) corr with true {names[b]} {truth[names[b]]['corr']:+.2f};"
                    " y-acc per (y, " + names[b] + "): "
                    + " ".join(f"{a:.2f}" for a in acc.values())
                )
            entry["truth"] = truth
        entry["vector"] = [round(float(x), 6) for x in dirs[j]]
        entries.append(entry)

    out = dict(
        run_dir=args.run_dir,
        ckpt=args.ckpt,
        clip_model=cfg.clip_model,
        prompted=[list(p) for p in pairs],
        directions=entries,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCore directions written to {args.out} (pooled accepted); "
          "train with --core_dirs and --lambda_promote")


if __name__ == "__main__":
    main()
