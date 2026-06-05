"""
losses.py — Combined loss for concept-debiased representation learning.

Three terms
-----------
L_task     : cross-entropy  — preserves downstream classification accuracy
L_align    : cosine distance between P(E(x)) and V_I_perp
             — pulls the projected representation toward the scrubbed target
L_repulse  : cosine similarity between P(E(x)) and V_T
             — explicitly pushes away from the concept direction
             (Note: if V_I_perp is well-estimated, L_align already implies
             low concept alignment.  L_repulse adds an explicit gradient
             signal and is weighted lower by default.)

Total loss = λ_task · L_task + λ_align · L_align + λ_repulse · L_repulse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Individual loss terms ─────────────────────────────────────────────────────


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy classification loss."""
    return F.cross_entropy(logits, targets)


def alignment_loss(
    proj: torch.Tensor,
    v_i_perp: torch.Tensor,
) -> torch.Tensor:
    """
    1 - cosine_similarity(P(E(x)), V_I_perp), averaged over the batch.

    Both inputs are expected to be (approximately) unit-normalised.
    Cosine similarity ∈ [-1, 1], so the loss ∈ [0, 2].
    """
    # cosine_similarity returns (B,)
    cos_sim = F.cosine_similarity(proj, v_i_perp, dim=-1)
    return (1.0 - cos_sim).mean()


def repulsion_loss(
    proj: torch.Tensor,
    v_t_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise cosine alignment of P(E(x)) with the concept direction V_T.

    We use the *squared* cosine similarity to penalise both positive and
    negative correlations with the concept direction (the representation
    should be truly orthogonal, not merely anti-correlated).

        L_repulse = mean( (P(E(x)) · V_T_hat)^2 )

    Parameters
    ----------
    proj    : (B, D) unit-normalised projected embeddings
    v_t_hat : (D,)   unit concept direction
    """
    # dot product per sample: (B,)
    dot = (proj * v_t_hat).sum(dim=-1)
    return (dot**2).mean()


# ── Combined loss ─────────────────────────────────────────────────────────────


class DebiasingLoss(nn.Module):
    """
    Wraps all three terms with configurable weights.

    Usage
    -----
        criterion = DebiasingLoss(cfg)
        loss, breakdown = criterion(logits, labels, proj, v_i_perp, v_t_hat)
    """

    def __init__(self, cfg):
        super().__init__()
        self.lambda_task = cfg.lambda_task
        self.lambda_align = cfg.lambda_align
        self.lambda_repulse = cfg.lambda_repulse

    def forward(
        self,
        logits: torch.Tensor,  # (B, num_classes)
        labels: torch.Tensor,  # (B,)
        proj: torch.Tensor,  # (B, D)  — P(E(x)), unit-normalised
        v_i_perp: torch.Tensor | None = None,  # (B, D)  — distillation target
        v_t_hat: torch.Tensor | None = None,  # (D,)    — concept direction
    ):
        """
        Returns
        -------
        total : scalar loss to call .backward() on
        info  : dict of individual (unweighted) loss values for logging

        Notes
        -----
        v_i_perp and v_t_hat may be None when the corresponding lambda is 0
        (ERM mode).  The zero-weight terms are skipped entirely to avoid
        unnecessary computation and to keep ERM a true baseline.
        """
        l_task = task_loss(logits, labels)

        l_align = (
            alignment_loss(proj, v_i_perp)
            if (self.lambda_align > 0 and v_i_perp is not None)
            else torch.tensor(0.0)
        )
        l_repulse = (
            repulsion_loss(proj, v_t_hat)
            if (self.lambda_repulse > 0 and v_t_hat is not None)
            else torch.tensor(0.0)
        )

        total = (
            self.lambda_task * l_task
            + self.lambda_align * l_align
            + self.lambda_repulse * l_repulse
        )

        info = {
            "loss_task": l_task.item(),
            "loss_align": l_align.item(),
            "loss_repulse": l_repulse.item(),
            "loss_total": total.item(),
        }
        return total, info
