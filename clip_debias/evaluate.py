"""
evaluate.py — Post-hoc evaluation of debiased representations.

Three evaluations
-----------------
1. Linear probe accuracy for the concept (gender) on E(x).
   A drop toward chance (50 %) indicates the concept has been scrubbed.

2. Downstream task accuracy (overall top-1).
   Should stay high after debiasing.

3. Worst-group accuracy — accuracy sliced by (target × concept) subgroup.
   The minimum across the 4 groups is the standard CelebA fairness metric.
   ERM typically has a large best-to-worst gap; debiasing should close it.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model,
    loader,
    device: str,
    use_amp: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run the model in eval mode and collect:
        embeds   : (N, embed_dim)  penultimate activations E(x)
        projs    : (N, clip_dim)   projected embeddings P(E(x))
        preds    : (N,)            argmax predictions
        targets  : (N,)            task labels
        concepts : (N,)            concept labels (gender for CelebA)
    """
    model.eval()
    model.to(device)

    all_embeds, all_projs, all_preds, all_targets, all_concepts = [], [], [], [], []

    for images, labels, concepts in loader:
        images = images.to(device)
        with autocast(enabled=use_amp):
            out = model(images)

        all_embeds.append(out["embed"].cpu().float())
        all_projs.append(out["proj"].cpu().float())
        all_preds.append(out["logits"].argmax(dim=1).cpu())
        all_targets.append(labels)
        all_concepts.append(concepts)

    return {
        "embeds":   torch.cat(all_embeds).numpy(),
        "projs":    torch.cat(all_projs).numpy(),
        "preds":    torch.cat(all_preds).numpy(),
        "targets":  torch.cat(all_targets).numpy(),
        "concepts": torch.cat(all_concepts).numpy(),
    }


# ── Linear probe ─────────────────────────────────────────────────────────────

def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """
    Fit a logistic regression probe on frozen embeddings.
    Features are z-scored before fitting (standard for linear probing).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
    clf.fit(X_train_s, y_train)

    return {
        "probe_train_acc": accuracy_score(y_train, clf.predict(X_train_s)),
        "probe_test_acc":  accuracy_score(y_test,  clf.predict(X_test_s)),
    }


# ── Worst-group accuracy ──────────────────────────────────────────────────────

def worst_group_accuracy(
    targets:  np.ndarray,
    concepts: np.ndarray,
    preds:    np.ndarray,
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """
    Compute per-group and worst-group accuracy over (target × concept) cells.

    For CelebA with target=Attractive and concept=Male the four groups are:
        (0, 0)  not attractive, female
        (0, 1)  not attractive, male
        (1, 0)  attractive, female
        (1, 1)  attractive, male

    Returns
    -------
    group_accs : dict mapping (target_val, concept_val) → accuracy
    worst_acc  : float — minimum accuracy across all groups
    """
    target_vals  = np.unique(targets)
    concept_vals = np.unique(concepts)

    group_accs: Dict[Tuple[int, int], float] = {}
    for t in target_vals:
        for c in concept_vals:
            mask = (targets == t) & (concepts == c)
            if mask.sum() == 0:
                continue
            acc = (preds[mask] == targets[mask]).mean()
            group_accs[(int(t), int(c))] = float(acc)

    worst_acc = min(group_accs.values())
    return group_accs, worst_acc


# ── Task accuracy (overall) ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_task(model, loader, device: str, use_amp: bool = True) -> float:
    """Return top-1 task accuracy over the loader."""
    model.eval()
    model.to(device)
    correct = total = 0
    for images, labels, _concepts in loader:
        images = images.to(device)
        labels = labels.to(device)
        with autocast(enabled=use_amp):
            out = model(images)
        correct += (out["logits"].argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Full evaluation suite ─────────────────────────────────────────────────────

def run_evaluation(
    model,
    train_loader,
    test_loader,
    device: str,
    use_amp: bool = True,
    label: str = "model",
) -> Dict[str, float]:
    """
    Run all three evaluations and print a summary table.

    Returns
    -------
    metrics : dict with keys
        probe_train_acc, probe_test_acc,
        task_test_acc,
        worst_group_acc,
        group_acc_(t,c) for each (target, concept) cell
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"{'='*60}")

    print("  Extracting train features …")
    train_feats = extract_features(model, train_loader, device, use_amp)

    print("  Extracting test features …")
    test_feats  = extract_features(model, test_loader,  device, use_amp)

    # ── 1. Concept probe ─────────────────────────────────────────────────────
    print("  Fitting concept probe on E(x) …")
    probe_metrics = train_linear_probe(
        X_train=train_feats["embeds"], y_train=train_feats["concepts"],
        X_test=test_feats["embeds"],   y_test=test_feats["concepts"],
    )

    # ── 2. Overall task accuracy ──────────────────────────────────────────────
    task_acc = float((test_feats["preds"] == test_feats["targets"]).mean())

    # ── 3. Worst-group accuracy ───────────────────────────────────────────────
    group_accs, worst_acc = worst_group_accuracy(
        targets=test_feats["targets"],
        concepts=test_feats["concepts"],
        preds=test_feats["preds"],
    )

    # ── Assemble & print ──────────────────────────────────────────────────────
    metrics: Dict[str, float] = {
        **probe_metrics,
        "task_test_acc":  task_acc,
        "worst_group_acc": worst_acc,
        **{f"group_acc_{k}": v for k, v in group_accs.items()},
    }

    print(f"\n  Results for '{label}':")
    print(f"    Concept probe test acc  : {probe_metrics['probe_test_acc']:.4f}"
          f"  (chance = 0.50)")
    print(f"    Overall task acc (test) : {task_acc:.4f}")
    print(f"    Worst-group acc (test)  : {worst_acc:.4f}")
    print(f"    Per-group breakdown:")
    for (t, c), acc in sorted(group_accs.items()):
        print(f"      target={t}, concept={c}  →  {acc:.4f}"
              f"  (n={int(((test_feats['targets']==t)&(test_feats['concepts']==c)).sum())})")

    return metrics