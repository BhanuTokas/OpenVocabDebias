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
         an early checkpoint (--ckpt epoch_01.pt), or hold examples out: pass
         the run directories of one scheme, either the K runs of a K-fold split
         (training flags --n_folds K --fold k --fold_seed s) or any number of
         resampled runs (--resample subsample|bootstrap --resample_frac f
         --resample_seed r).  Each train example is predicted only by the runs
         that held it out; it is an error if most of them get it wrong (a tie
         is not), and examples no run held out are left out of the directions.
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

from clip_debias.baselines import split_indices
from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_multi_concept_subspace,
    concept_margin,
    embed_dataset,
)
from clip_debias.config import get_concept_prompts
from clip_debias.data import build_dataloaders
from clip_debias.discovery import (
    as_prompt,
    direction_scores,
    error_directions,
    load_vocabulary,
    name_scores,
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
    p.add_argument(
        "--vocab_file",
        nargs="+",
        default=None,
        help="one-phrase-per-line files whose union replaces the built-in "
        "vocabulary (e.g. prompts/vocab/); see also --objects",
    )
    p.add_argument(
        "--clip_cache",
        default=None,
        help="file holding the split's CLIP embeddings: read if it exists, else "
        "written, so repeated calls on one split skip the CLIP image pass",
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


def prompt_scores(codes, c, y):
    """How well zero-shot codes label concept c.  Plain accuracy rewards labelling
    the class wherever c correlates with y, so also average over (y, c) cells and
    report the worst cell, in the code orientation with the better average."""
    cells = [(y == a) & (c == b) for a in np.unique(y) for b in np.unique(c)]
    acc = [float((codes[m] == c[m]).mean()) for m in cells]
    if np.mean(acc) < 0.5:
        codes, acc = ~codes, [1 - a for a in acc]
    return dict(
        prompt_acc=round(float((codes == c).mean()), 3),
        prompt_bal=round(float(np.mean(acc)), 3),
        prompt_worst=round(float(min(acc)), 3),
    )


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

    held_out = cfg.n_folds > 0 or bool(cfg.resample)
    if len(runs) > 1 or held_out:
        scheme = (cfg.n_folds, cfg.fold_seed, cfg.resample, cfg.resample_frac)
        K = cfg.n_folds
        ids = [c.resample_seed if c.resample else c.fold for c, _ in runs]
        if (
            args.split != "train"
            or not held_out
            or any(
                (c.n_folds, c.fold_seed, c.resample, c.resample_frac) != scheme
                for c, _ in runs
            )
            or len(set(ids)) != len(ids)
            or (K > 0 and sorted(ids) != list(range(K)))
        ):
            raise SystemExit(
                "held-out errors need --split train and the runs of one scheme: "
                "every fold 0..K-1 of one split (--n_folds K --fold k --fold_seed s) "
                "or distinct draws (--resample m --resample_frac f --resample_seed r)"
            )
        y, wrong, votes = np.full(n, -1), np.zeros(n, int), np.zeros(n, int)
        for d, (c, _) in zip(args.run_dir, runs):
            idx = split_indices(n, c)[1]
            y[idx], e = model_errors(d, c, ds, idx, args)
            wrong[idx] += e
            votes[idx] += 1
        err, covered = 2 * wrong > votes, votes > 0
        name = cfg.resample or f"{K}-fold"
        source = (
            f"held-out train ({name}, {len(runs)} runs, {covered.mean():.0%} of train)"
        )
    else:
        y, err = model_errors(args.run_dir[0], cfg, ds, np.arange(n), args)
        covered = np.ones(n, bool)
        source = args.split
    print(
        f"\n{cfg.run_name}  seed={cfg.seed}  {args.ckpt}: {source} error "
        f"{err[covered].mean():.3f}"
    )
    for k in np.unique(y[covered]):
        m = covered & (y == k)
        print(f"  class {k}: {int((err & m).sum())} errors / {int(m.sum())}")

    oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
    if args.clip_cache and os.path.isfile(args.clip_cache):
        v, labels, c_true = torch.load(args.clip_cache)
    else:
        v, labels, c_true = embed_dataset(oracle, ds, num_workers=args.num_workers)
        if args.clip_cache:
            torch.save((v, labels, c_true), args.clip_cache)
    y_all = labels.numpy().astype(int)
    pairs = prompted_pairs(cfg, dataset) if args.prompted == "config" else []
    S = (
        compute_multi_concept_subspace(oracle, pairs, k=cfg.concept_subspace_k)
        if pairs
        else None
    )
    if pairs:
        print(f"  removing {len(pairs)} prompted concept(s), subspace {tuple(S.shape)}")

    dirs, classes = error_directions(v[covered], y[covered], err[covered])
    z = direction_scores(v, S, dirs)
    vocab = (
        load_vocabulary(args.vocab_file)
        if args.vocab_file
        else vocabulary(objects=args.objects)
    )
    print(f"  vocabulary: {len(vocab)} phrases")
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
                corr_y=round(float(np.corrcoef(z[:, j], y_all)[0, 1]), 3),
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
            # how well the named phrases work as prompts, not just the direction
            codes = concept_margin(oracle, entry["pos"], entry["neg"], v) > 0
            entry.update(prompt_scores(codes.numpy(), c_true.numpy() & 1, y_all))
            print(
                f"    (analysis) prompts label true {names[0]}: accuracy "
                f"{entry['prompt_acc']:.3f}, balanced over (class, concept) "
                f"{entry['prompt_bal']:.3f}, worst cell {entry['prompt_worst']:.3f}"
            )
        proposals.append(entry)

    out = dict(
        run_dir=args.run_dir if len(args.run_dir) > 1 else args.run_dir[0],
        coverage=round(float(covered.mean()), 4),
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
        n_errors=np.array([int((err & covered & (y == k)).sum()) for k in classes]),
    )
    print(
        f'\nProposals written to {args.out}: review, edit, set "accept": true, '
        "then train with --extra_prompts"
    )


if __name__ == "__main__":
    main()
