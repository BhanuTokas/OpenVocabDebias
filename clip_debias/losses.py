"""
losses.py — Loss functions for dual-optimizer debiasing.

Two separate loss groups, each targeting a different set of parameters:

Proj head loss (L_reconstruct)
-------------------------------
L_reconstruct : 1 - cos(P(E(x)), V_I)
    The projection head's only job is to reconstruct the original CLIP image
    embedding as faithfully as possible.  No concept geometry here — that is
    entirely the backbone's responsibility.

Backbone loss (L_backbone)
---------------------------
L_task     : cross-entropy — preserve downstream classification accuracy
L_align    : 1 - cos(P(E(x)), V_I_perp) — push P(E(x)) toward the
             concept-scrubbed target.  Because the proj head is a faithful
             CLIP reconstructor, the only way to satisfy this is for E(x)
             itself to lose the concept component.
L_repulse  : (P(E(x)) · V̂_T)² — directly penalise concept alignment in
             the projected space, with gradient flowing only into the backbone.

Total backbone loss = λ_task·L_task + λ_align·L_align + λ_repulse·L_repulse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Individual loss terms ─────────────────────────────────────────────────────


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy classification loss."""
    return F.cross_entropy(logits, targets)


def reconstruction_loss(
    proj: torch.Tensor,
    v_i: torch.Tensor,
) -> torch.Tensor:
    """
    Proj head loss: 1 - cos(P(E(x)), V_I).

    Encourages the projection head to reconstruct the original (un-scrubbed)
    CLIP image embedding.  Both inputs should be unit-normalised.
    """
    return (1.0 - F.cosine_similarity(proj, v_i, dim=-1)).mean()


def alignment_loss(
    proj: torch.Tensor,
    v_i_perp: torch.Tensor,
) -> torch.Tensor:
    """
    Backbone loss: 1 - cos(P(E(x)), V_I_perp).

    Pushes the backbone to produce embeddings that, when projected by the
    (reconstruction-trained) proj head, land near the concept-scrubbed target.
    Both inputs should be unit-normalised.
    """
    return (1.0 - F.cosine_similarity(proj, v_i_perp, dim=-1)).mean()


def repulsion_loss(
    proj: torch.Tensor,
    v_t_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Backbone loss: mean( (P(E(x)) · V̂_T)² ).

    Penalises both positive and negative concept alignment in the projected
    space.  Gradient flows only into the backbone (proj head is frozen for
    this term via the dual-optimizer setup in trainer.py).
    """
    dot = (proj * v_t_hat).sum(dim=-1)
    return (dot**2).mean()


# ── Loss group wrappers ───────────────────────────────────────────────────────


class ProjHeadLoss(nn.Module):
    """
    Loss for the projection head optimizer.
    Reconstruction of the full CLIP image embedding V_I.
    """

    def forward(
        self,
        proj: torch.Tensor,  # (B, D) P(E(x)), unit-normalised
        v_i: torch.Tensor,  # (B, D) full CLIP image embedding, unit-normalised
    ):
        l = reconstruction_loss(proj, v_i)
        return l, {"loss_reconstruct": l.item()}


class BackboneLoss(nn.Module):
    """
    Loss for the backbone optimizer.
    Task + alignment toward V_I_perp + repulsion from V̂_T.
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
        proj: torch.Tensor,  # (B, D) P(E(x)), unit-normalised
        v_i_perp: torch.Tensor | None = None,  # (B, D) concept-scrubbed target
        v_t_hat: torch.Tensor | None = None,  # (D,)   unit concept direction
    ):
        l_task = task_loss(logits, labels)

        l_align = (
            alignment_loss(proj, v_i_perp)
            if (self.lambda_align > 0 and v_i_perp is not None)
            else logits.new_zeros(())
        )
        l_repulse = (
            repulsion_loss(proj, v_t_hat)
            if (self.lambda_repulse > 0 and v_t_hat is not None)
            else logits.new_zeros(())
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
            "loss_backbone": total.item(),
        }
        return total, info
