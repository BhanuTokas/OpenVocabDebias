"""
clip_oracle.py — CLIP as a zero-shot concept oracle.

Key responsibilities
--------------------
1. Load and freeze the CLIP model.
2. Build the concept SUBSPACE from a prompt ensemble via SVD.
3. Compute concept-scrubbed CLIP image embeddings V_I_perp.

Subspace vs single direction
-----------------------------
The original single-direction approach computed:
    V̂_T = normalise( mean(pos_embeds) - mean(neg_embeds) )   shape: (D,)

The subspace approach stacks per-pair difference vectors and decomposes them:
    diff_i = normalise(pos_i - neg_i)                         shape: (P, D)
    _, _, Vt = svd(diff_matrix)
    subspace = Vt[:k]                                         shape: (k, D)

Each row of `subspace` is an orthonormal basis vector.  Projecting out all k
directions removes a richer representation of the concept than a single vector.
k=1 recovers a result close to the original single-direction behaviour.

Verification signal
-------------------
After switching to subspace, `loss_align` and `loss_reconstruct` should
diverge in the training logs.  If they still track each other, increase k.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ── CLIP wrapper ──────────────────────────────────────────────────────────────


class CLIPOracle:
    """Thin wrapper around a frozen HuggingFace CLIP model."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """(N, D) unit-normalised text embeddings."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, D) unit-normalised image embeddings from CLIP-preprocessed pixels."""
        out = self.model.get_image_features(pixel_values=pixel_values)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_images_from_pil(self, pil_images) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)


# ── Concept subspace (replaces single direction) ──────────────────────────────


def compute_concept_subspace(
    oracle: CLIPOracle,
    prompts_pos: List[str],
    prompts_neg: List[str],
    k: int = 3,
) -> torch.Tensor:
    """
    Compute a k-dimensional orthonormal subspace spanning the concept directions.

    Algorithm
    ---------
    1. Encode each prompt independently (not averaged).
    2. Compute per-pair difference vectors: diff_i = normalise(pos_i - neg_i).
       If len(pos) != len(neg), use the mean of the shorter list as the other pole.
    3. Stack into a matrix and extract the top-k right singular vectors via SVD.

    Parameters
    ----------
    prompts_pos : positive-pole prompts (e.g. male descriptions)
    prompts_neg : negative-pole prompts (e.g. female descriptions)
    k           : number of subspace dimensions.  k=1 ≈ single-direction behaviour.
                  Start with k=3; ablate k=1,2,3,5.

    Returns
    -------
    subspace : (k, D) tensor — each row is a unit-norm basis vector.
               Rows are orthonormal (guaranteed by SVD).
    """
    with torch.no_grad():
        pos_embeds = oracle.encode_text(prompts_pos)  # (P, D)
        neg_embeds = oracle.encode_text(prompts_neg)  # (N, D)

        # Align lengths: broadcast the mean of the shorter list
        if len(prompts_pos) == len(prompts_neg):
            diff_vectors = F.normalize(pos_embeds - neg_embeds, dim=-1)  # (P, D)
        elif len(prompts_pos) > len(prompts_neg):
            neg_mean = neg_embeds.mean(dim=0, keepdim=True)  # (1, D)
            diff_vectors = F.normalize(pos_embeds - neg_mean, dim=-1)  # (P, D)
        else:
            pos_mean = pos_embeds.mean(dim=0, keepdim=True)  # (1, D)
            diff_vectors = F.normalize(neg_embeds - pos_mean, dim=-1)  # (N, D)

        # SVD: diff_vectors = U @ S @ Vt
        # Vt rows are the right singular vectors — the principal directions
        # of variance in the difference matrix.
        _, _, Vt = torch.linalg.svd(diff_vectors, full_matrices=False)
        subspace = Vt[:k]  # (k, D) — already orthonormal from SVD

    k_actual = subspace.shape[0]
    if k_actual < k:
        print(
            f"  [compute_concept_subspace] WARNING: requested k={k} but only "
            f"{k_actual} singular vectors available (P={diff_vectors.shape[0]}). "
            f"Using k={k_actual}."
        )

    print(
        f"  [compute_concept_subspace] k={k_actual}  "
        f"subspace shape={tuple(subspace.shape)}  "
        f"orthonormality check (should be I): "
        f"max|SS^T - I|={((subspace @ subspace.T) - torch.eye(k_actual, device=subspace.device)).abs().max().item():.2e}"
    )
    return subspace


# ── Kept for backwards compatibility / Waterbirds use ────────────────────────


def build_concept_direction(
    oracle: CLIPOracle,
    prompts_pos: List[str],
    prompts_neg: List[str],
) -> torch.Tensor:
    """
    Original single-direction approach.  Equivalent to compute_concept_subspace(k=1)
    but returns a (D,) vector rather than (1, D).

    Kept for backwards compatibility with waterbirds_train.py.
    New code should use compute_concept_subspace().
    """
    v_pos = oracle.encode_text(prompts_pos).mean(dim=0)
    v_neg = oracle.encode_text(prompts_neg).mean(dim=0)
    return F.normalize(v_pos - v_neg, dim=0)  # (D,)


# ── Orthogonal projection ─────────────────────────────────────────────────────


def orthogonal_project_subspace(
    v_i: torch.Tensor,
    subspace: torch.Tensor,
) -> torch.Tensor:
    """
    Remove all components of V_I that lie within the concept subspace.

    Applies sequential 1-D projections along each basis vector.
    This is correct because SVD rows are orthonormal — projecting out
    one basis vector does not affect the others, so order doesn't matter.

        for each basis vector b in subspace:
            v_i = v_i - (v_i · b) * b

    Parameters
    ----------
    v_i      : (B, D) or (..., D)  CLIP image embeddings
    subspace : (k, D)              orthonormal basis of the concept subspace

    Returns
    -------
    v_i_perp : (..., D)  concept-scrubbed embeddings, NOT re-normalised.
    """
    for i in range(subspace.shape[0]):
        b = subspace[i]  # (D,)
        coeff = (v_i * b).sum(dim=-1, keepdim=True)  # (..., 1)
        v_i = v_i - coeff * b
    return v_i


def orthogonal_project(
    v_i: torch.Tensor,
    v_t_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Original single-direction projection.  Kept for backwards compatibility.
    New code should use orthogonal_project_subspace().
    """
    coeff = (v_i * v_t_hat).sum(dim=-1, keepdim=True)
    return v_i - coeff * v_t_hat


# ── Distillation target computation ──────────────────────────────────────────


@torch.no_grad()
def compute_distillation_targets(
    oracle: CLIPOracle,
    pixel_values: torch.Tensor,
    subspace: torch.Tensor,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full pipeline for one batch:
      1. Encode images with frozen CLIP  →  V_I
      2. Project out concept subspace    →  V_I_perp
      3. Optionally re-normalise         →  unit V_I_perp

    Parameters
    ----------
    oracle       : CLIPOracle instance
    pixel_values : (B, 3, H, W) CLIP-preprocessed pixel values
    subspace     : (k, D) orthonormal concept subspace
                   (pass a (1, D) tensor to replicate the single-direction case)
    renormalize  : whether to L2-normalise V_I_perp before returning

    Returns
    -------
    v_i      : (B, D)  original CLIP image embeddings (proj head target)
    v_i_perp : (B, D)  concept-scrubbed embeddings    (backbone target)
    """
    v_i = oracle.encode_images(pixel_values)  # (B, D)
    v_i_perp = orthogonal_project_subspace(v_i, subspace)  # (B, D)
    if renormalize:
        v_i_perp = F.normalize(v_i_perp, dim=-1)
    return v_i, v_i_perp
