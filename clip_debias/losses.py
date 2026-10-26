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
L_repulse  (repulse="whitened"):
             sum_k mean_j corr_batch(E(x)_j, z_k)^2, where z = whitened CLIP
             concept scores of the same images (clip_oracle.whitened_concept_scores).
             Zero cross-covariance between E(x) and z means no linear probe on
             E(x) predicts z better than a constant — the LEACE condition,
             imposed on the backbone features instead of the target.
L_repulse  (repulse="whitened_cond"):
             the same, but E(x) and z are centred within each class first, so
             it penalises corr(E(x), z | y).  Under a spurious correlation the
             concept and the label are confounded; the marginal version then
             also removes label information, the conditional one does not.
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


def decorrelation_loss(
    embed: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Squared batch correlation between every feature of E(x) and every concept score.

        L = sum_k mean_j corr(embed_j, scores_k)^2

    Parameters
    ----------
    embed  : (B, d) backbone features E(x)
    scores : (B, r) whitened CLIP concept scores (no gradient)
    labels : (B,) class labels.  If given, embed and scores are centred within
             each class, giving the partial correlation corr(embed_j, scores_k | y)
             pooled over classes.

    Under independence each corr^2 is ~1/B, so the floor is about r/B.
    """
    with torch.autocast(embed.device.type, enabled=False):
        e = embed.float()
        z = scores.float()
        if labels is None:
            e = e - e.mean(0)
            z = z - z.mean(0)
        else:
            onehot = F.one_hot(labels.long()).float()  # (B, C)
            counts = onehot.sum(0).clamp_min(1.0)[:, None]  # (C, 1)
            e = e - onehot @ (onehot.T @ e / counts)
            z = z - onehot @ (onehot.T @ z / counts)
        corr = (e.T @ z) / (e.norm(dim=0)[:, None] * z.norm(dim=0)[None, :] + eps)
        return corr.pow(2).mean(0).sum()


def promotion_loss(
    pred: torch.Tensor, scores: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Promote core concepts: mean_k (1 − corr(pred_k, scores_k)^2) over the batch.

    pred   : (B, r) linear read-out of E(x) (a head trained with the backbone)
    scores : (B, r) whitened core-concept scores (no gradient), residual to the
             prompted concepts, so E(x) is pushed to encode what the core
             directions carry beyond the bias.  Sign-free, scale-free.
    """
    with torch.autocast(pred.device.type, enabled=False):
        p = pred.float() - pred.float().mean(0)
        z = scores.float() - scores.float().mean(0)
        corr = (p * z).sum(0) / (p.norm(dim=0) * z.norm(dim=0) + eps)
        return (1.0 - corr.pow(2)).mean()


def cka_alignment_loss(embed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 − linear CKA between backbone features E(x) (B, d) and a target
    representation of the same images (B, D), e.g. the concept-erased CLIP
    embedding.  CKA compares the centred batch Gram matrices, so it needs no
    projection head and is invariant to rotations and isotropic scaling of
    either side.
    """
    with torch.autocast(embed.device.type, enabled=False):
        e = embed.float() - embed.float().mean(0)
        t = target.float() - target.float().mean(0)
        k, l = e @ e.T, t @ t.T
        cka = (k * l).sum() / (k.norm() * l.norm() + 1e-8)
        return 1.0 - cka


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
        self.whitened_repulse = cfg.repulse in ("whitened", "whitened_cond", "raw")
        self.conditional_repulse = cfg.repulse == "whitened_cond"
        self.cka_align = cfg.align == "cka"
        self.lambda_promote = cfg.lambda_promote

    def forward(
        self,
        logits: torch.Tensor,  # (B, num_classes)
        labels: torch.Tensor,  # (B,)
        proj: torch.Tensor,  # (B, D) unit-normalised
        v_i_perp: torch.Tensor | None = None,  # (B, D) concept-scrubbed target
        subspace: torch.Tensor | None = None,  # (k, D) concept subspace
        # Legacy single-direction argument kept for backwards compatibility:
        v_t_hat: torch.Tensor | None = None,  # (D,) — auto-converted to (1, D)
        embed: torch.Tensor | None = None,  # (B, d) E(x), for repulse="whitened"
        concept_scores: torch.Tensor | None = None,  # (B, r) whitened CLIP scores
        core_pred: torch.Tensor | None = None,  # (B, r_core) head read-out of E(x)
        core_scores: torch.Tensor | None = None,  # (B, r_core) whitened core scores
    ):
        l_task = task_loss(logits, labels)

        if self.lambda_align > 0 and v_i_perp is not None:
            l_align = (
                cka_alignment_loss(embed, v_i_perp)
                if self.cka_align
                else alignment_loss(proj, v_i_perp)
            )
        else:
            l_align = logits.new_zeros(())

        # Resolve subspace: prefer explicit subspace, fall back to v_t_hat
        _subspace = subspace
        if _subspace is None and v_t_hat is not None:
            _subspace = v_t_hat.unsqueeze(0)  # (D,) → (1, D)

        if self.lambda_repulse <= 0:
            l_repulse = logits.new_zeros(())
        elif self.whitened_repulse:
            if embed is None or concept_scores is None:
                raise ValueError("repulse='whitened' needs embed and concept_scores")
            l_repulse = decorrelation_loss(
                embed, concept_scores, labels if self.conditional_repulse else None
            )
        elif _subspace is not None:
            l_repulse = repulsion_loss(proj, _subspace)
        else:
            l_repulse = logits.new_zeros(())

        if self.lambda_promote > 0 and core_pred is not None:
            l_promote = promotion_loss(core_pred, core_scores)
        else:
            l_promote = logits.new_zeros(())

        total = (
            self.lambda_task * l_task
            + self.lambda_align * l_align
            + self.lambda_repulse * l_repulse
            + self.lambda_promote * l_promote
        )

        info = {
            "loss_task": l_task.item(),
            "loss_align": l_align.item(),
            "loss_repulse": l_repulse.item(),
            "loss_promote": l_promote.item(),
            "loss_backbone": total.item(),
        }
        return total, info
