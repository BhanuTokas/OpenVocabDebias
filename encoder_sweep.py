#!/usr/bin/env python
"""
encoder_sweep.py — Compare frozen image-text encoders (CLIP, SigLIP, ...) on the
concept signals the debiasing method relies on, without training.

For each encoder and concept it reports, on the val split:
  zs_acc     CLIP zero-shot concept accuracy (drives the pseudo-groups and the
             label-free checkpoint selection)
  corr_z0    |corr| between the first whitened concept score and the true
             concept (drives the decorrelation loss; the eraser is fit on train
             embeddings exactly as in training)
Waterbirds place (word and sentence prompts) and CelebA Male, Young and the
joint Male+Young code are covered.  Images go through the training data
pipeline and the encoder's own preprocessing (oracle.preprocess).
Embeddings are cached per encoder in --out_dir.

Usage
-----
    python encoder_sweep.py --celeba_root C:/path/to/CelebA
    python encoder_sweep.py --encoders google/siglip-base-patch16-224 --skip_celeba
    python encoder_sweep.py --encoders hf-hub:timm/PE-Core-L-14-336 --skip_celeba  # OpenCLIP
"""

import argparse
import csv
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clip_debias.clip_oracle import (
    CLIPOracle,
    compute_concept_subspace,
    fit_leace_eraser,
    whitened_concept_scores,
    zeroshot_concept_labels,
)
from clip_debias.config import erm_config, get_concept_prompts
from clip_debias.data import build_dataloaders
from clip_debias.waterbirds_data import build_waterbirds_dataloaders
from waterbirds_train import PROMPT_SETS, WaterbirdsConfig

ENCODERS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-224",
    "google/siglip2-base-patch16-224",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "hf-hub:timm/PE-Core-B-16",  # Perception Encoder (OpenCLIP port)
    "hf-hub:timm/PE-Core-L-14-336",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--encoders", nargs="+", default=ENCODERS)
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--skip_celeba", action="store_true")
    p.add_argument("--celeba_n_train", type=int, default=20000)
    p.add_argument("--celeba_n_val", type=int, default=10000)
    p.add_argument("--k", type=int, default=9, help="text directions per concept (capped by prompt pairs)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="./results/encoder_sweep")
    return p.parse_args()


@torch.no_grad()
def embed(oracle, dataset, n, args, seed=0):
    """CLIP embeddings and (label, concept) for a fixed random subset."""
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n]
    loader = DataLoader(Subset(dataset, idx.tolist()), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    V, Y, C = [], [], []
    for x, y, c in loader:
        V.append(oracle.encode_images(oracle.preprocess(x.to(oracle.device))).float().cpu())
        Y.append(y)
        C.append(c)
    return torch.cat(V), torch.cat(Y), torch.cat(C)


def cached(path, fn):
    if os.path.isfile(path):
        z = np.load(path)
        return tuple(torch.from_numpy(z[k]) for k in ("v", "y", "c"))
    v, y, c = fn()
    np.savez(path, v=v.numpy(), y=y.numpy(), c=c.numpy())
    return v, y, c


def concept_metrics(oracle, pos, neg, k, v_tr, v_va, c_va):
    zs = zeroshot_concept_labels(oracle, pos, neg, v_va)
    S = compute_concept_subspace(oracle, pos, neg, k=min(k, len(pos), len(neg)))
    z = whitened_concept_scores(v_va, fit_leace_eraser(v_tr, S))
    corr = abs(float(np.corrcoef(z[:, 0].numpy(), c_va.numpy())[0, 1]))
    return float((zs == c_va).float().mean()), corr, zs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "encoder_sweep.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wb = build_waterbirds_dataloaders(WaterbirdsConfig(num_workers=0))
    cb = None
    if not args.skip_celeba:
        if args.celeba_root is None:
            raise SystemExit("--celeba_root is required unless --skip_celeba")
        cb = build_dataloaders(erm_config(celeba_root=args.celeba_root, target_attr="Blond_Hair",
                                          concept_attr="Male", extra_concept_attrs=["Young"],
                                          num_workers=0))
    rows = []
    for name in args.encoders:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            oracle = CLIPOracle(name, device=device)
        except Exception as e:  # missing weights, unsupported architecture, ...
            print(f"  [skip] could not load: {type(e).__name__}: {e}")
            continue
        tag = re.sub(r"[^\w.-]", "_", name)
        base = dict(encoder=name, embed_dim=oracle.embed_dim, input_px=oracle.image_size)

        tr = cached(os.path.join(args.out_dir, f"{tag}_wb_train.npz"),
                    lambda: embed(oracle, wb[0].dataset, len(wb[0].dataset), args))
        va = cached(os.path.join(args.out_dir, f"{tag}_wb_val.npz"),
                    lambda: embed(oracle, wb[1].dataset, len(wb[1].dataset), args))
        for pset in ("words", "sentences"):
            acc, corr, _ = concept_metrics(oracle, *PROMPT_SETS[pset], 3, tr[0], va[0], va[2])
            rows.append(dict(base, dataset="waterbirds", concept="place", prompts=pset,
                             zs_acc=acc, corr_z0=corr))

        if cb is not None:
            tr = cached(os.path.join(args.out_dir, f"{tag}_celeba_train.npz"),
                        lambda: embed(oracle, cb[0].dataset, args.celeba_n_train, args))
            va = cached(os.path.join(args.out_dir, f"{tag}_celeba_val.npz"),
                        lambda: embed(oracle, cb[1].dataset, args.celeba_n_val, args))
            joint = torch.zeros_like(va[2])
            for bit, attr in enumerate(("Male", "Young")):
                c_bit = (va[2] >> bit) & 1
                acc, corr, zs = concept_metrics(oracle, *get_concept_prompts(attr), args.k,
                                                tr[0], va[0], c_bit)
                joint += zs << bit
                rows.append(dict(base, dataset="celeba", concept=attr, prompts="library",
                                 zs_acc=acc, corr_z0=corr))
            rows.append(dict(base, dataset="celeba", concept="Male+Young", prompts="library",
                             zs_acc=float((joint == va[2]).float().mean()), corr_z0=""))
        print(f"  done in {time.time() - t0:.0f}s")
        for r in rows[-5:]:
            if r["encoder"] == name:
                print(f"  {r['dataset']:10s} {r['concept']:10s} {r['prompts']:9s} "
                      f"zs_acc {r['zs_acc']:.3f}  corr_z0 {r['corr_z0'] if r['corr_z0'] == '' else round(r['corr_z0'], 3)}")
        del oracle
        torch.cuda.empty_cache()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {out_csv}")
    print(f"\n{'encoder':<42}" + "".join(f"{h:<14}" for h in
          ["WB words", "WB sent.", "Male", "Young", "Male+Young"]) + "  (zero-shot acc / |corr z0|)")
    for name in dict.fromkeys(r["encoder"] for r in rows):
        rs = [r for r in rows if r["encoder"] == name]
        cell = lambda r: f"{r['zs_acc']:.3f}" + (f"/{r['corr_z0']:.2f}" if r["corr_z0"] != "" else "")
        print(f"{name:<42}" + "".join(f"{cell(r):<14}" for r in rs))


if __name__ == "__main__":
    main()
