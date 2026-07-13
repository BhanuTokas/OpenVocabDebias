"""
losses.py — Loss functions for dual-optimizer debiasing.

Two separate loss groups:

Proj head loss (L_reconstruct)
-------------------------------
L_reconstruct : 1 - cos(P(E(x)), V_I)
    Projection head reconstructs the ORIGINAL (un-scrubbed) CLIP image
    embedding.  No concept geometry here.

Backbone loss (L_backbone)
---------------------------
L_task     : cross-entropy
L_align    : 1 - cos(P(E(x)), V_I_perp)
             V_I_perp is scrubbed by projecting out the full concept subspace,
             so this term forces the backbone to remove ALL concept directions.
L_repulse  : mean( sum_k (P(E(x)) · b_k)^2 )  where b_k are the subspace bases.
             Penalises alignment with every direction in the concept subspace.
             Gradient flows only into the backbone via the dual-optimizer setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Individual loss terms ─────────────────────────────────────────────────────


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def reconstruction_loss(proj: torch.Tensor, v_i: torch.Tensor) -> torch.Tensor:
    """1 - cos(P(E(x)), V_I).  Both inputs should be unit-normalised."""
    return (1.0 - F.cosine_similarity(proj, v_i, dim=-1)).mean()


def alignment_loss(proj: torch.Tensor, v_i_perp: torch.Tensor) -> torch.Tensor:
    """1 - cos(P(E(x)), V_I_perp).  Both inputs should be unit-normalised."""
    return (1.0 - F.cosine_similarity(proj, v_i_perp, dim=-1)).mean()


def repulsion_loss(
    proj: torch.Tensor,
    subspace: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise alignment of P(E(x)) with every direction in the concept subspace.

        L_repulse = mean_batch( sum_{k} (proj · b_k)^2 )

    Parameters
    ----------
    proj     : (B, D) unit-normalised projected embeddings
    subspace : (k, D) orthonormal concept subspace basis

    Notes
    -----
    The sum over k means that all concept directions are penalised jointly,
    not just the primary one.  For k=1 this reduces to the original scalar
    repulsion: mean( (proj · v̂_T)^2 ).
    """
    projections = proj @ subspace.T  # (B, k)
    return projections.pow(2).sum(dim=-1).mean()


# ── Loss group wrappers ───────────────────────────────────────────────────────


class ProjHeadLoss(nn.Module):
    """Reconstruction of the full CLIP image embedding V_I."""

    def forward(
        self,
        proj: torch.Tensor,  # (B, D) P(E(x)), unit-normalised
        v_i: torch.Tensor,  # (B, D) full CLIP image embedding, unit-normalised
    ):
        l = reconstruction_loss(proj, v_i)
        return l, {"loss_reconstruct": l.item()}


class BackboneLoss(nn.Module):
    """
    Task + alignment toward V_I_perp + repulsion from concept subspace.

    v_t_hat / subspace
    ------------------
    Accepts either:
      subspace : (k, D)  — full k-dimensional concept subspace  [recommended]
      v_t_hat  : (D,)    — single concept direction (legacy; auto-converted to (1, D))
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
        proj: torch.Tensor,  # (B, D) unit-normalised
        v_i_perp: torch.Tensor | None = None,  # (B, D) concept-scrubbed target
        subspace: torch.Tensor | None = None,  # (k, D) concept subspace
        # Legacy single-direction argument kept for backwards compatibility:
        v_t_hat: torch.Tensor | None = None,  # (D,) — auto-converted to (1, D)
    ):
        l_task = task_loss(logits, labels)

        l_align = (
            alignment_loss(proj, v_i_perp)
            if (self.lambda_align > 0 and v_i_perp is not None)
            else logits.new_zeros(())
        )

        # Resolve subspace: prefer explicit subspace, fall back to v_t_hat
        _subspace = subspace
        if _subspace is None and v_t_hat is not None:
            _subspace = v_t_hat.unsqueeze(0)  # (D,) → (1, D)

        l_repulse = (
            repulsion_loss(proj, _subspace)
            if (self.lambda_repulse > 0 and _subspace is not None)
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
