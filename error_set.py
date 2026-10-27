#!/usr/bin/env python
"""
error_set.py — Train-set error set of an early checkpoint, for JTT / BAM stage 2.

Loads a run's checkpoint (e.g. epoch_01.pt of an ERM run for JTT, or of a
--bam_aux run for BAM; the BAM auxiliaries are not part of the model, so they
are removed automatically), predicts every train image without augmentation,
and writes {err, y, c} aligned with the train dataset indices.  Training with
--error_set <file> --upweight U then upsamples the errors U times.

Only target labels are used; the true concept c is stored for analysis and the
per-group error rates are printed (reported only).

Usage
-----
    python error_set.py <run dir> --ckpt epoch_01.pt --out jtt_wb_seed42_T1.npz
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from clip_debias.data import CelebADebias, get_eval_transform
from clip_debias.evaluate import extract_features
from clip_debias.models import build_model
from clip_debias.waterbirds_data import WaterbirdsDebiasDataset
from eval_checkpoints import load_run_cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="run directory containing config.json and the checkpoint")
    p.add_argument("--ckpt", default="epoch_01.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg, dataset, _ = load_run_cfg(args.run_dir, SimpleNamespace(**vars(args), dataset="auto"))
    if dataset == "waterbirds":
        ds = WaterbirdsDebiasDataset("train", transform=get_eval_transform())
    else:
        ds = CelebADebias(
            cfg.celeba_root,
            "train",
            cfg.target_attr,
            cfg.concept_attr,
            extra_concept_attrs=cfg.extra_concept_attrs,
            transform=get_eval_transform(),
        )
    model = build_model(cfg, num_classes=2)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=cfg.device)
    model.load_state_dict(state.get("model_state", state))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    f = extract_features(model, loader, cfg.device, use_amp=cfg.amp)
    y, c = f["targets"].astype(int), f["concepts"].astype(int)
    err = f["preds"] != f["targets"]

    print(f"{cfg.run_name}  seed={cfg.seed}  {args.ckpt}: {err.sum():,} / {len(err):,} "
          f"train errors ({err.mean():.3f})")
    print("  (analysis) error rate per true (y, concept) group: " + "  ".join(
        f"({a},{b}) {err[(y == a) & (c == b)].mean():.2f} n={((y == a) & (c == b)).sum()}"
        for a in np.unique(y) for b in np.unique(c)
    ))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, err=err, y=y, c=c)
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
