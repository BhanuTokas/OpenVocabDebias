"""
clip_oracle.py — Everything related to CLIP as a zero-shot concept oracle.

Key responsibilities
--------------------
1. Load and freeze the CLIP model.
2. Build the concept direction V_T from a prompt ensemble.
3. Compute concept-scrubbed CLIP image embeddings V_I_perp.
4. Cache image embeddings across the dataset to avoid re-computing.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ── CLIP wrapper ──────────────────────────────────────────────────────────────


class CLIPOracle:
    """
    Thin wrapper around a frozen HuggingFace CLIP model.

    All tensors are returned on the same device as the model.
    The model's weights are frozen and set to eval mode; it is
    never updated during debiasing training.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze everything
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ── Text encoding ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of strings → (N, D) unit-normalised text embeddings.
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats, dim=-1)

    # ── Image encoding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images (already CLIP-preprocessed as pixel_values)
        → (B, D) unit-normalised image embeddings.

        If you pass raw ImageNet-normalised tensors use encode_images_raw()
        which handles re-normalisation internally.
        """
        out = self.model.get_image_features(pixel_values=pixel_values)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_images_from_pil(self, pil_images) -> torch.Tensor:
        """Encode a list of PIL images."""
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)


# ── Concept direction ─────────────────────────────────────────────────────────


def build_concept_direction(
    oracle: CLIPOracle,
    prompts_pos: List[str],
    prompts_neg: List[str],
) -> torch.Tensor:
    """
    Compute a unit-normalised concept direction in CLIP text space.

    Strategy: directional difference between two prompt poles.
        V_T = normalise( mean(CLIP_text(pos)) - mean(CLIP_text(neg)) )

    This is more robust than averaging all prompts together because it
    captures a *direction* (e.g. male→female) rather than a point.

    Returns
    -------
    v_t : (D,) unit vector on the CLIP text embedding sphere.
    """
    v_pos = oracle.encode_text(prompts_pos).mean(dim=0)  # (D,)
    v_neg = oracle.encode_text(prompts_neg).mean(dim=0)  # (D,)
    direction = v_pos - v_neg
    return F.normalize(direction, dim=0)  # (D,)


def build_concept_direction_average(
    oracle: CLIPOracle,
    prompts: List[str],
) -> torch.Tensor:
    """
    Fallback: average all prompts and normalise.
    Use when you don't have natural positive/negative poles.

    Returns
    -------
    v_t : (D,) unit vector.
    """
    embeddings = oracle.encode_text(prompts)  # (N, D)
    mean_embed = embeddings.mean(dim=0)  # (D,)
    return F.normalize(mean_embed, dim=0)


# ── Orthogonal projection ─────────────────────────────────────────────────────


def orthogonal_project(
    v_i: torch.Tensor,
    v_t_hat: torch.Tensor,
) -> torch.Tensor:
    """
    Remove the component of V_I along V_T_hat (concept direction).

        V_I_perp = V_I - (V_I · V_T_hat) * V_T_hat

    Parameters
    ----------
    v_i     : (..., D)  CLIP image embeddings (batch or single vector).
    v_t_hat : (D,)      Unit-normalised concept direction.

    Returns
    -------
    v_i_perp : (..., D) concept-scrubbed embeddings, NOT re-normalised.
               Callers that need unit vectors should normalise afterwards.

    Notes
    -----
    * v_t_hat must already be unit-normalised (the function does NOT
      re-normalise it to avoid silent correctness bugs).
    * The output is NOT on the unit sphere.  Whether to re-normalise is a
      design choice: re-normalising makes the distillation target easier to
      match via cosine loss but slightly changes the geometry.  We leave
      it to the caller.
    """
    # scalar projection: (..., 1)
    coeff = (v_i * v_t_hat).sum(dim=-1, keepdim=True)
    return v_i - coeff * v_t_hat


# ── Convenience: compute targets for a batch ─────────────────────────────────


@torch.no_grad()
def compute_distillation_targets(
    oracle: CLIPOracle,
    pixel_values: torch.Tensor,
    v_t_hat: torch.Tensor,
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Full pipeline for one batch:
      1. Encode images with frozen CLIP  →  V_I
      2. Project out concept direction   →  V_I_perp
      3. Optionally re-normalise         →  unit V_I_perp

    Parameters
    ----------
    oracle       : CLIPOracle instance
    pixel_values : (B, 3, H, W) — CLIP-preprocessed pixel values
    v_t_hat      : (D,) unit concept direction
    renormalize  : whether to L2-normalise V_I_perp before returning

    Returns
    -------
    targets : (B, D) distillation targets for the alignment loss
    """
    v_i = oracle.encode_images(pixel_values)  # (B, D)
    v_i_perp = orthogonal_project(v_i, v_t_hat)  # (B, D)
    if renormalize:
        v_i_perp = F.normalize(v_i_perp, dim=-1)
    return v_i_perp
