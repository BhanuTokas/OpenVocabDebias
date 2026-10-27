#!/usr/bin/env python
"""
clip_probe.py — Frozen-CLIP baselines: what the CLIP oracle alone achieves.

Rows written (one per method, --out CSV):
  clip_zeroshot        class-name prompts, no training
  clip_probe           logistic regression on frozen CLIP image embeddings
  clip_probe_balanced  the same, train examples weighted to balance the
                       (target x CLIP zero-shot concept) pseudo-groups of the
                       concept prompts (label-free, as grw_pseudo)
C is selected by worst-group accuracy on val pseudo-groups (label-free, as
best_pseudo_wga.pt); test metrics use the true groups.  Embeddings use the
eval transform (no augmentation) and are cached in --cache_dir.

Usage
-----
    python clip_probe.py --dataset waterbirds --prompts sentences --out probe_wb.csv
    python clip_probe.py --dataset celeba --celeba_root /path/to/CelebA \\
        --concept_attr Male --extra_concept_attrs Young --out probe_celeba.csv
"""

import argparse
import csv
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from clip_debias.clip_oracle import (
    CLIPOracle,
    concept_margin,
    embed_dataset,
    zeroshot_concept_codes,
)
from clip_debias.config import get_concept_prompts
from clip_debias.data import CelebADebias, get_eval_transform
from clip_debias.discovery import load_prompt_file
from clip_debias.reweight import group_weights
from clip_debias.waterbirds_data import WaterbirdsDebiasDataset

CLASS_PROMPTS = {
    "waterbirds": (["a photo of a waterbird"], ["a photo of a landbird"]),
    "celeba": (
        ["a photo of a person with blond hair", "a photo of a blond person"],
        ["a photo of a person with dark hair", "a photo of a person with brown hair"],
    ),
}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["waterbirds", "celeba"], required=True)
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--prompts", default="sentences", help="Waterbirds prompt set")
    p.add_argument("--extra_prompts", default=None, help="discovered prompt file (replaces --prompts)")
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--target_attr", default="Blond_Hair")
    p.add_argument("--concept_attr", default="Male")
    p.add_argument("--extra_concept_attrs", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--cache_dir", default="./results/clip_probe")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def datasets(args):
    t = get_eval_transform()
    if args.dataset == "waterbirds":
        return {s: WaterbirdsDebiasDataset(n, transform=t)
                for s, n in (("train", "train"), ("val", "validation"), ("test", "test"))}
    return {s: CelebADebias(args.celeba_root, n, args.target_attr, args.concept_attr,
                            extra_concept_attrs=args.extra_concept_attrs, transform=t)
            for s, n in (("train", "train"), ("val", "valid"), ("test", "test"))}


def prompt_pairs(args):
    if args.extra_prompts:
        return load_prompt_file(args.extra_prompts)
    if args.dataset == "waterbirds":
        from waterbirds_train import PROMPT_SETS

        return [PROMPT_SETS[args.prompts]]
    return [get_concept_prompts(a) for a in [args.concept_attr] + args.extra_concept_attrs]


def wga(pred, y, g):
    return min(float((pred[g == k] == y[g == k]).mean()) for k in np.unique(g))


def main():
    args = parse_args()
    oracle = CLIPOracle(args.clip_model, device="cuda")
    tag = re.sub(r"[^\w.-]", "_", args.clip_model)
    attrs = "" if args.dataset == "waterbirds" else "_" + "-".join([args.concept_attr] + args.extra_concept_attrs)
    data = {}
    for split, ds in datasets(args).items():
        path = os.path.join(args.cache_dir, f"{tag}_{args.dataset}{attrs}_{split}.npz")
        if not os.path.isfile(path):
            v, y, c = embed_dataset(oracle, ds, num_workers=args.num_workers)
            os.makedirs(args.cache_dir, exist_ok=True)
            np.savez(path, v=v.numpy(), y=y.numpy(), c=c.numpy())
        d = np.load(path)
        data[split] = (torch.tensor(d["v"]).float(), d["y"].astype(int), d["c"].astype(int))
        print(f"  {split}: {len(d['y']):,} embeddings ({path})")

    pairs = prompt_pairs(args)
    n_attr = 1 + len(args.extra_concept_attrs) if args.dataset == "celeba" else 1
    n_codes = 1 << n_attr
    (vtr, ytr, ctr), (vva, yva, cva), (vte, yte, cte) = data["train"], data["val"], data["test"]
    chat_tr = zeroshot_concept_codes(oracle, pairs, vtr).numpy().astype(int)
    chat_va = zeroshot_concept_codes(oracle, pairs, vva).numpy().astype(int)
    print(f"  pseudo-group accuracy (reported only): train {(chat_tr == ctr).mean():.3f}  "
          f"val {(chat_va == cva).mean():.3f}")
    g_va, g_te = yva * n_codes + chat_va, yte * n_codes + cte

    def row(name, pred, params):
        r = dict(
            run_name=name,
            dataset=args.dataset,
            clip_model=args.clip_model,
            prompts=args.extra_prompts or (args.prompts if args.dataset == "waterbirds" else attrs[1:]),
            task_test_acc=round(float((pred == yte).mean()), 4),
            worst_group_acc=round(wga(pred, yte, g_te), 4),
            **{f"wga_attr{b}": round(wga(pred, yte, yte * 2 + ((cte >> b) & 1)), 4) for b in range(n_attr)},
            params=params,
        )
        print(f"  {name:<20} test acc {r['task_test_acc']:.3f}  WGA {r['worst_group_acc']:.3f}  {params}")
        return r

    rows = []
    pos, neg = CLASS_PROMPTS[args.dataset]
    rows.append(row("clip_zeroshot", (concept_margin(oracle, pos, neg, vte).numpy() > 0).astype(int), {}))
    bal = group_weights(torch.tensor(ytr), torch.tensor(chat_tr)).numpy()
    bal = bal * len(bal) / bal.sum()  # mean 1, so C means the same as unweighted
    for name, sw in (("clip_probe", None), ("clip_probe_balanced", bal)):
        best = (-1.0, None, None)
        for C in C_GRID:
            lr = LogisticRegression(C=C, max_iter=2000).fit(vtr.numpy(), ytr, sample_weight=sw)
            score = wga(lr.predict(vva.numpy()), yva, g_va)
            if score > best[0]:
                best = (score, C, lr)
        rows.append(row(name, best[2].predict(vte.numpy()),
                        dict(C=best[1], val_pseudo_wga=round(best[0], 4))))

    new = not os.path.isfile(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"  appended to {args.out}")


if __name__ == "__main__":
    main()
