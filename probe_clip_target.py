#!/usr/bin/env python
"""
probe_clip_target.py — Is the CLIP distillation target V_I_perp concept-free?

Diagnostic, no training.  Encodes every split with frozen CLIP (same ImageNet →
CLIP renormalisation as training, eval transform), removes the concept with
several erasers, and measures what is left:

    concept_v2t   concept probe fit on val, tested on test (0.5 = nothing left;
                  on Waterbirds val/test, place is independent of the label)
    concept_t2t   concept probe fit on train, tested on test (the convention
                  used by evaluate.py / eval_checkpoints.py)
    task_acc,     task classifier fit on train, tested on test — does the
    task_wga      erased target help worst-group accuracy?
    energy        mean squared norm of V_I inside the removed subspace
    var_ratio     mean variance of V_I along the removed directions, relative
                  to the average per-dimension variance (1.0 = like a random
                  direction; orthogonal erasers only)

Erasers
-------
    orth:<prompts>      orthogonal projection of the text SVD subspace (what
                        training does), label-free
    leace:<prompts>     whitened (LEACE-style) erasure of everything linearly
                        correlated with the text-subspace scores; uses only the
                        unlabeled image covariance, label-free
    leace:zeroshot      LEACE on CLIP zero-shot concept pseudo-labels,
                        label-free
    random              random orthonormal subspace (control)
    oracle:*            uses true concept labels (train split) — reference only

Usage
-----
    python probe_clip_target.py --dataset waterbirds
    python probe_clip_target.py --dataset celeba --celeba_root <root> \\
        --target_attr Blond_Hair --concept_attr Male
"""

import argparse
import csv
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from clip_debias.clip_oracle import CLIPOracle, compute_concept_subspace, fit_leace
from clip_debias.clip_preprocess import RenormalizeForCLIP
from clip_debias.config import DebiasingConfig, get_concept_prompts
from clip_debias.data import CelebADebias, get_eval_transform
from clip_debias.waterbirds_data import WaterbirdsDebiasDataset
from waterbirds_train import NEG_WORDS, POS_WORDS

SPLITS = ["train", "val", "test"]

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

MALE_NOUNS = ["man", "male person", "boy", "gentleman", "guy", "father"]
FEMALE_NOUNS = ["woman", "female person", "girl", "lady", "gal", "mother"]
FACE_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of the face of a {}",
    "a portrait of a {}",
    "a celebrity photo of a {}",
]


def prompt_sets(args) -> Dict[str, Tuple[List[str], List[str]]]:
    if args.dataset == "waterbirds":
        return {
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
    sets = {"library": get_concept_prompts(args.concept_attr)}
    if args.concept_attr == "Male":
        sets["ensemble"] = (
            [t.format(n) for t in FACE_TEMPLATES for n in MALE_NOUNS],
            [t.format(n) for t in FACE_TEMPLATES for n in FEMALE_NOUNS],
        )
    return sets


def parse_args():
    cfg = DebiasingConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--dataset", default="waterbirds", choices=["waterbirds", "celeba"]
    )
    parser.add_argument("--celeba_root", default=cfg.celeba_root)
    parser.add_argument("--target_attr", default="Blond_Hair")
    parser.add_argument("--concept_attr", default="Male")
    parser.add_argument(
        "--max_train",
        type=int,
        default=30000,
        help="Random subset of the train split (CelebA train is 162k)",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--clip_model", default=cfg.clip_model)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 8, 13, 24])
    parser.add_argument("--inlp_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = (
            "./results_waterbirds/clip_target_probe"
            if args.dataset == "waterbirds"
            else f"./results/clip_target_probe/{args.target_attr}_{args.concept_attr}"
        )
    return args


# ── Embeddings ────────────────────────────────────────────────────────────────


def build_split(args, split: str):
    tf = get_eval_transform()
    if args.dataset == "waterbirds":
        return WaterbirdsDebiasDataset(
            {"train": "train", "val": "validation", "test": "test"}[split], transform=tf
        )
    ds = CelebADebias(
        args.celeba_root,
        {"train": "train", "val": "valid", "test": "test"}[split],
        args.target_attr,
        args.concept_attr,
        transform=tf,
    )
    if split == "train" and len(ds) > args.max_train:
        idx = np.random.default_rng(args.seed).choice(
            len(ds), args.max_train, replace=False
        )
        ds = Subset(ds, np.sort(idx).tolist())
    return ds


def encode_splits(oracle: CLIPOracle, args) -> Dict[str, Dict[str, np.ndarray]]:
    """V_I, task label y and concept label c per split; cached as .npz in out_dir."""
    cache = os.path.join(
        args.out_dir,
        f"clip_embeds_{args.clip_model.replace('/', '_')}"
        f"_train{args.max_train}_seed{args.seed}.npz",
    )
    if os.path.isfile(cache):
        print(f"Loading cached embeddings from {cache}")
        z = np.load(cache)
        return {s: {k: z[f"{s}_{k}"] for k in ("v_i", "y", "c")} for s in SPLITS}

    renorm = RenormalizeForCLIP().to(oracle.device)
    data, flat = {}, {}
    for split in SPLITS:
        loader = DataLoader(
            build_split(args, split),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        v_i, ys, cs = [], [], []
        for images, y, c in loader:
            v_i.append(oracle.encode_images(renorm(images.to(oracle.device))).cpu())
            ys.append(y)
            cs.append(c)
        data[split] = {
            "v_i": torch.cat(v_i).float().numpy(),
            "y": torch.cat(ys).numpy(),
            "c": torch.cat(cs).numpy(),
        }
        print(f"  encoded {split}: {data[split]['v_i'].shape}")
        flat.update({f"{split}_{k}": v for k, v in data[split].items()})
    np.savez(cache, **flat)
    return data


# ── Erasers ───────────────────────────────────────────────────────────────────
# Each eraser maps raw V_I (N, D) → erased, re-normalised embeddings (as the
# training target is re-normalised after projection).


def _renorm(x: np.ndarray) -> np.ndarray:
    return F.normalize(torch.from_numpy(x.astype(np.float32)), dim=-1).numpy()


def orth_eraser(basis: np.ndarray) -> Callable:
    """Orthogonal projection onto the complement of an orthonormal (k, D) basis."""
    return lambda x: _renorm(x - (x @ basis.T) @ basis) if len(basis) else x


def leace_eraser(x_fit: np.ndarray, cross_cov: np.ndarray) -> Callable:
    """Whitened (LEACE-style) eraser — same fit as training (clip_oracle.fit_leace)."""
    mu, M, _ = (
        t.double().numpy()
        for t in fit_leace(torch.from_numpy(x_fit), torch.from_numpy(cross_cov))
    )
    return lambda x: _renorm(x - (x - mu) @ M.T)


def text_basis(oracle, pos, neg, k_max) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal text subspace (k, D) and the singular values of the diff matrix."""
    with torch.no_grad():
        p, n = oracle.encode_text(pos), oracle.encode_text(neg)
        if len(pos) == len(neg):
            diffs = F.normalize(p - n, dim=-1)
        elif len(pos) > len(neg):
            diffs = F.normalize(p - n.mean(0, keepdim=True), dim=-1)
        else:
            diffs = F.normalize(n - p.mean(0, keepdim=True), dim=-1)
        sv = torch.linalg.svdvals(diffs).cpu().numpy()
    basis = compute_concept_subspace(oracle, pos, neg, k=k_max).cpu().float().numpy()
    return basis, sv


def zeroshot_labels(oracle, pos, neg, x: np.ndarray) -> np.ndarray:
    """CLIP zero-shot concept prediction: closer to the mean pos or mean neg prompt."""
    with torch.no_grad():
        d = (
            (oracle.encode_text(pos).mean(0) - oracle.encode_text(neg).mean(0))
            .cpu()
            .numpy()
        )
    return (x @ d > 0).astype(int)


def inlp_bases(x: np.ndarray, c: np.ndarray, iters: int, seed: int) -> List[np.ndarray]:
    """Iterative nullspace projection on concept labels."""
    basis = np.zeros((0, x.shape[1]), dtype=np.float64)
    out = []
    for _ in range(iters):
        xp = x - (x @ basis.T) @ basis
        w = LogisticRegression(max_iter=2000, random_state=seed).fit(xp, c).coef_[0]
        w = w - basis.T @ (basis @ w)
        basis = np.vstack([basis, w / np.linalg.norm(w)])
        out.append(basis.copy())
    return out


# ── Probes ────────────────────────────────────────────────────────────────────


def fit_predict(X_train, y_train, X_test, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    return clf.predict(scaler.transform(X_test))


def worst_group(preds, y, c) -> float:
    return min(
        (preds[m] == y[m]).mean()
        for a in (0, 1)
        for b in (0, 1)
        if (m := (y == a) & (c == b)).any()
    )


def evaluate(data, erase: Callable, seed: int) -> Dict[str, float]:
    tr, va, te = (data[s] for s in SPLITS)
    X = {s: erase(data[s]["v_i"]) for s in SPLITS}
    task_pred = fit_predict(X["train"], tr["y"], X["test"], seed)
    return {
        "concept_v2t": float(
            (fit_predict(X["val"], va["c"], X["test"], seed) == te["c"]).mean()
        ),
        "concept_t2t": float(
            (fit_predict(X["train"], tr["c"], X["test"], seed) == te["c"]).mean()
        ),
        "task_acc": float((task_pred == te["y"]).mean()),
        "task_wga": float(worst_group(task_pred, te["y"], te["c"])),
    }


def basis_stats(x: np.ndarray, basis: np.ndarray) -> Dict[str, float]:
    if not len(basis):
        return {"energy": 0.0, "var_ratio": ""}
    per_dim = np.var(x, axis=0).sum() / x.shape[1]
    proj = x @ basis.T
    return {
        "energy": float((proj**2).sum(1).mean()),
        "var_ratio": float(proj.var(0).mean() / per_dim),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nLoading CLIP ({args.clip_model}) …")
    oracle = CLIPOracle(args.clip_model, device=device)

    data = encode_splits(oracle, args)
    x_tr, c_tr = data["train"]["v_i"].astype(np.float64), data["train"]["c"]
    dim = x_tr.shape[1]
    ks = sorted(set(args.ks))

    # (source, k, label_free, eraser, basis-or-None)
    configs = [("none", 0, True, lambda x: x, np.zeros((0, dim)))]
    md = x_tr[c_tr == 1].mean(0) - x_tr[c_tr == 0].mean(0)
    cov = np.cov(x_tr, rowvar=False)

    print("\nPrompt-set diagnostics")
    for name, (pos, neg) in prompt_sets(args).items():
        full, sv = text_basis(oracle, pos, neg, max(ks))
        share = sv**2 / (sv**2).sum()
        zs = zeroshot_labels(oracle, pos, neg, data["test"]["v_i"])
        print(
            f"  {name:<10} pairs={len(sv):>2}  top-5 variance share of diffs: "
            f"{np.round(share[:5], 3).tolist()}"
        )
        print(
            f"  {'':<10} |cos(top text dir, image mean-diff)| = "
            f"{abs(full[0] @ md) / np.linalg.norm(md):.3f}   "
            f"zero-shot concept acc (test) = {(zs == data['test']['c']).mean():.3f}"
        )
        for k in sorted({min(k, len(full)) for k in ks}):
            b = full[:k].astype(np.float64)
            configs.append((f"orth:{name}", k, True, orth_eraser(b), b))
            # cross-covariance of x with its own text-subspace scores = Σ bᵀ
            configs.append(
                (f"leace:{name}", k, True, leace_eraser(x_tr, cov @ b.T), None)
            )
        z = zeroshot_labels(oracle, pos, neg, x_tr)
        if 0 < z.mean() < 1:
            cc = ((x_tr - x_tr.mean(0)) * (z - z.mean())[:, None]).mean(0)[:, None]
            configs.append(
                (f"leace:zeroshot_{name}", 1, True, leace_eraser(x_tr, cc), None)
            )

    rng = np.random.default_rng(args.seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, max(ks))))
    for k in ks:
        configs.append(("random", k, True, orth_eraser(q.T[:k]), q.T[:k]))

    md_basis = (md / np.linalg.norm(md))[None]
    configs.append(("oracle:meandiff", 1, False, orth_eraser(md_basis), md_basis))
    cc = ((x_tr - x_tr.mean(0)) * (c_tr - c_tr.mean())[:, None]).mean(0)[:, None]
    configs.append(("oracle:leace", 1, False, leace_eraser(x_tr, cc), None))
    for b in inlp_bases(x_tr, c_tr, args.inlp_iters, args.seed):
        if len(b) in (1, 2, 3, 5, 10, 20) or len(b) == args.inlp_iters:
            configs.append(("oracle:inlp", len(b), False, orth_eraser(b), b))

    rows = []
    hdr = (
        f"\n{'source':<24}{'k':>4}{'energy':>9}{'var_ratio':>10}{'concept_v2t':>13}"
        f"{'concept_t2t':>13}{'task_acc':>10}{'task_wga':>10}"
    )
    print(hdr)
    for source, k, label_free, erase, basis in configs:
        stats = (
            basis_stats(x_tr, basis)
            if basis is not None
            else {"energy": "", "var_ratio": ""}
        )
        m = evaluate(data, erase, args.seed)
        rows.append({"source": source, "k": k, "label_free": label_free, **stats, **m})
        fmt = lambda v, w: f"{v:>{w}.4f}" if isinstance(v, float) else f"{'-':>{w}}"
        print(
            f"{source:<24}{k:>4}{fmt(stats['energy'], 9)}{fmt(stats['var_ratio'], 10)}"
            f"{m['concept_v2t']:>13.4f}{m['concept_t2t']:>13.4f}"
            f"{m['task_acc']:>10.4f}{m['task_wga']:>10.4f}"
        )

    out_csv = os.path.join(args.out_dir, "probe_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
