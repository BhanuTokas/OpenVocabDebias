#!/usr/bin/env python
"""
last_layer.py — Last-layer retraining baselines on a trained (ERM) model.

  dfr  DFR (Kirichenko et al., ICLR 2023, DFR^Val_Tr): standardised features,
       L1 logistic regression on group-balanced subsamples of the val split.
       C is tuned on val half A -> worst-group accuracy on half B; the final
       model averages 20 fits on balanced subsamples of the whole val split.
  afr  AFR (Qiu et al., ICML 2023): last layer initialised at the ERM one and
       retrained by full-batch gradient descent on
           sum_i mu_i CE_i + lambda ||phi - phi_ERM||^2,
       mu_i ∝ beta_{y_i} exp(-gamma p_i) (p_i: ERM probability of the true
       class, beta_y: 1 / class size).  AFR uses no group labels in training.
       Adaptation: AFR's reweighting set is a held-out 20% of train with the
       ERM model trained on the other 80%; here the ERM model saw all of train,
       so val half A is the reweighting set (as DFR uses val), and (gamma,
       lambda, step) are selected on half B.

--groups decides which groups balance (DFR) and select (both) on val:
  true    dataset concept labels (reference: uses bias annotations)
  pseudo  CLIP zero-shot codes of the run's concept prompts (label-free, the
          same pseudo-groups as best_pseudo_wga.pt)
Test metrics always use the true groups.  One CSV row per call (--out).

Usage
-----
    python last_layer.py <ERM run dir> --method dfr --groups pseudo --out ll.csv
"""

import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clip_debias.clip_oracle import CLIPOracle, embed_dataset, zeroshot_concept_codes
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import extract_features
from clip_debias.models import build_model
from clip_debias.waterbirds_data import build_waterbirds_dataloaders
from discover_biases import prompted_pairs
from eval_checkpoints import load_run_cfg

DFR_C = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
AFR_GRID = {
    # gamma grid (fewer points than the paper's 33 / 10), lambda grid, lr, epochs
    "waterbirds": (np.linspace(4, 20, 9), [0.0, 0.1, 0.2, 0.3, 0.4], 1e-2, 500),
    "celeba": (np.linspace(1, 3, 5), [0.001, 0.01, 0.1], 2e-2, 1000),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="ERM run directory (config.json + checkpoint)")
    p.add_argument(
        "--ckpt", default="best.pt", help="feature extractor (default: val-acc)"
    )
    p.add_argument("--method", choices=["dfr", "afr"], required=True)
    p.add_argument("--groups", choices=["true", "pseudo"], required=True)
    p.add_argument("--seed", type=int, default=0, help="val halves and subsampling")
    p.add_argument("--celeba_root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def wga(pred, y, g):
    accs = {int(k): float((pred[g == k] == y[g == k]).mean()) for k in np.unique(g)}
    return min(accs.values()), accs


def balanced_subsample(g, rng):
    cells = [np.flatnonzero(g == k) for k in np.unique(g)]
    n = min(len(c) for c in cells)
    return np.concatenate([rng.choice(c, n, replace=False) for c in cells])


def fit_dfr(x, y, g, C, n_fits, rng):
    coefs, icpts = [], []
    for _ in range(n_fits):
        idx = balanced_subsample(g, rng)
        lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=1000)
        lr.fit(x[idx], y[idx])
        coefs.append(lr.coef_)
        icpts.append(lr.intercept_)
    return np.mean(coefs, 0), np.mean(icpts, 0)


def dfr(xv, yv, gv, A, B, xt, rng):
    sc = StandardScaler().fit(xv[A])
    best = (-1.0, None)
    for C in DFR_C:
        w, b = fit_dfr(sc.transform(xv[A]), yv[A], gv[A], C, 5, rng)
        pred = ((sc.transform(xv[B]) @ w.T + b).ravel() > 0).astype(int)
        score = wga(pred, yv[B], gv[B])[0]
        print(f"    C={C:<5} val-B WGA {score:.3f}")
        if score > best[0]:
            best = (score, C)
    sc = StandardScaler().fit(xv)
    w, b = fit_dfr(sc.transform(xv), yv, gv, best[1], 20, rng)
    pred = ((sc.transform(xt) @ w.T + b).ravel() > 0).astype(int)
    return pred, dict(C=best[1], val_select_wga=round(best[0], 4))


def afr(xv, yv, gv, A, B, xt, W0, b0, grid, device):
    gammas, lams, lr, epochs = grid
    # Features rescaled to unit mean norm and the ERM layer scaled to match
    # (same initial logits): with raw features (norm ~25 for R18) the fixed
    # normalised step of 1e-2 oscillates and the WGA collapses within 100 steps.
    s = float(np.linalg.norm(xv[A], axis=1).mean())
    xa, ya = torch.tensor(xv[A] / s, device=device), torch.tensor(yv[A], device=device)
    xb, xtt = torch.tensor(xv[B] / s, device=device), torch.tensor(
        xt / s, device=device
    )
    W0, b0 = (W0 * s).to(device), b0.to(device)
    with torch.no_grad():
        p = F.softmax(xa @ W0.T + b0, 1).gather(1, ya[:, None]).squeeze(1)
        beta = 1.0 / torch.bincount(ya).float()
    best = (-1.0, None, None)
    for gamma in gammas:
        mu = beta[ya] * torch.exp(-gamma * p)
        mu = mu / mu.sum()
        for lam in lams:
            W, b = W0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
            for step in range(1, epochs + 1):
                loss = (mu * F.cross_entropy(xa @ W.T + b, ya, reduction="none")).sum()
                loss = loss + lam * ((W - W0).pow(2).sum() + (b - b0).pow(2).sum())
                gW, gb = torch.autograd.grad(loss, [W, b])
                norm = torch.sqrt(gW.pow(2).sum() + gb.pow(2).sum()).clamp_min(1.0)
                with torch.no_grad():
                    W -= lr * gW / norm
                    b -= lr * gb / norm
                if step % 10 == 0 or step == epochs:
                    with torch.no_grad():
                        pred = (xb @ W.T + b).argmax(1).cpu().numpy()
                    score = wga(pred, yv[B], gv[B])[0]
                    if score > best[0]:
                        with torch.no_grad():
                            tpred = (xtt @ W.T + b).argmax(1).cpu().numpy()
                        best = (
                            score,
                            dict(gamma=round(float(gamma), 3), lam=lam, step=step),
                            tpred,
                        )
        print(f"    gamma={gamma:.2f}  best val-B WGA so far {best[0]:.3f}  {best[1]}")
    params = dict(best[1], val_select_wga=round(best[0], 4))
    return best[2], params


def main():
    args = parse_args()
    cfg, dataset, _ = load_run_cfg(
        args.run_dir, SimpleNamespace(**vars(args), dataset="auto")
    )
    _, val_l, test_l = (
        build_waterbirds_dataloaders(cfg)
        if dataset == "waterbirds"
        else build_dataloaders(cfg)
    )
    model = build_model(cfg, num_classes=2)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=cfg.device)
    model.load_state_dict(state.get("model_state", state))
    fv = extract_features(model, val_l, cfg.device, use_amp=cfg.amp)
    ft = extract_features(model, test_l, cfg.device, use_amp=cfg.amp)
    xv, yv, cv = (
        fv["embeds"].astype(np.float32),
        fv["targets"].astype(int),
        fv["concepts"].astype(int),
    )
    xt, yt, ct = (
        ft["embeds"].astype(np.float32),
        ft["targets"].astype(int),
        ft["concepts"].astype(int),
    )
    n_attr = 1 + len(cfg.extra_concept_attrs)
    n_codes = 1 << n_attr

    if args.groups == "true":
        cv_sel = cv
    else:
        oracle = CLIPOracle(cfg.clip_model, device=cfg.device)
        v, _, _ = embed_dataset(oracle, val_l.dataset, num_workers=args.num_workers)
        cv_sel = (
            zeroshot_concept_codes(
                oracle, prompted_pairs(cfg, dataset), v, cfg.zs_cutoffs or None
            )
            .numpy()
            .astype(int)
        )
        print(
            f"  pseudo-group accuracy on val (reported only): {(cv_sel == cv).mean():.3f}"
        )
    gv = yv * n_codes + cv_sel

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(yv))
    A, B = np.sort(perm[: len(yv) // 2]), np.sort(perm[len(yv) // 2 :])
    print(
        f"\n{cfg.run_name}  seed={cfg.seed}  {args.ckpt}  {args.method.upper()} "
        f"({args.groups} groups)  val {len(yv)} / test {len(yt)}"
    )
    if args.method == "dfr":
        pred, params = dfr(xv, yv, gv, A, B, xt, rng)
    else:
        fc = model.backbone.fc
        pred, params = afr(
            xv,
            yv,
            gv,
            A,
            B,
            xt,
            fc.weight.detach().float(),
            fc.bias.detach().float(),
            AFR_GRID[dataset],
            cfg.device if torch.cuda.is_available() else "cpu",
        )

    gt = yt * n_codes + ct
    worst, accs = wga(pred, yt, gt)
    row = dict(
        run_name=f"{args.method}_{args.groups}",
        source_run=cfg.run_name,
        seed=cfg.seed,
        dataset=dataset,
        backbone=cfg.backbone,
        ckpt=args.ckpt,
        task_test_acc=round(float((pred == yt).mean()), 4),
        worst_group_acc=round(worst, 4),
        **{
            f"wga_attr{b}": round(wga(pred, yt, yt * 2 + ((ct >> b) & 1))[0], 4)
            for b in range(n_attr)
        },
        params=params,
        group_accs={k: round(v, 4) for k, v in accs.items()},
    )
    print(f"  test acc {row['task_test_acc']:.3f}  WGA {worst:.3f}  {params}")
    new = not os.path.isfile(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  appended to {args.out}")


if __name__ == "__main__":
    main()
