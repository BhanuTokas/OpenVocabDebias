"""
evaluate.py — Post-hoc evaluation of debiased representations.

Two evaluations
---------------
1. Linear probe accuracy for the concept (gender) on E(x).
   Train a logistic regression on frozen embeddings before and after
   debiasing.  A drop toward chance (50 %) indicates the concept has
   been scrubbed from the representation.

2. Downstream task accuracy on the held-out test set.
   Standard top-1 accuracy; should stay high after debiasing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model,
    loader,
    device: str,
    use_amp: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run the model in eval mode over a DataLoader and collect:
        embeds   : (N, embed_dim)  — penultimate activations E(x)
        projs    : (N, clip_dim)   — projected embeddings P(E(x))
        targets  : (N,)            — task labels
        concepts : (N,)            — concept labels (for probe)

    Returns a dict of numpy arrays.
    """
    model.eval()
    model.to(device)

    all_embeds, all_projs, all_targets, all_concepts = [], [], [], []

    for images, labels, concepts in loader:
        images = images.to(device)
        with autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
            out = model(images)

        all_embeds.append(out["embed"].cpu().float())
        all_projs.append(out["proj"].cpu().float())
        all_targets.append(labels)
        all_concepts.append(concepts)

    return {
        "embeds":   torch.cat(all_embeds).numpy(),
        "projs":    torch.cat(all_projs).numpy(),
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
    Fit a logistic regression probe and report train + test accuracy.

    Features are z-scored before fitting (helps with convergence and is
    standard practice for linear probing).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))
    return {"probe_train_acc": train_acc, "probe_test_acc": test_acc}


# ── Downstream task accuracy ──────────────────────────────────────────────────

def task_accuracy(targets: np.ndarray, logits_or_embeds=None, model=None) -> float:
    """Compute top-1 accuracy from pre-collected target arrays."""
    raise NotImplementedError(
        "Pass targets and predicted labels directly via evaluate_model()."
    )


@torch.no_grad()
def evaluate_task(
    model,
    loader,
    device: str,
    use_amp: bool = True,
) -> float:
    """Return top-1 task accuracy over the loader."""
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    for images, labels, _concepts in loader:
        images = images.to(device)
        labels = labels.to(device)
        with autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp):
            out = model(images)
        correct += (out["logits"].argmax(1) == labels).sum().item()
        total += labels.size(0)

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
    Convenience function that runs both evaluations and prints a summary.

    Parameters
    ----------
    label : str
        A display name printed in the summary (e.g. "before" / "after").

    Returns
    -------
    metrics : dict with keys
        probe_train_acc, probe_test_acc, task_test_acc
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"{'='*60}")

    print("  Extracting train features …")
    train_feats = extract_features(model, train_loader, device, use_amp)

    print("  Extracting test features …")
    test_feats = extract_features(model, test_loader, device, use_amp)

    # ── Concept probe on E(x) ────────────────────────────────────────────────
    print("  Fitting concept probe on E(x) …")
    probe_metrics = train_linear_probe(
        X_train=train_feats["embeds"],
        y_train=train_feats["concepts"],
        X_test=test_feats["embeds"],
        y_test=test_feats["concepts"],
    )

    # ── Task accuracy ────────────────────────────────────────────────────────
    print("  Computing task accuracy …")
    task_acc = evaluate_task(model, test_loader, device, use_amp)

    metrics = {**probe_metrics, "task_test_acc": task_acc}

    print(f"\n  Results for '{label}':")
    print(f"    Concept probe (test acc) : {probe_metrics['probe_test_acc']:.4f}")
    print(f"    Task accuracy (test)     : {task_acc:.4f}")
    print(f"  (chance level for binary probe = 0.50)")

    return metrics
