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

from typing import List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

from .clip_preprocess import RenormalizeForCLIP

# ── CLIP wrapper ──────────────────────────────────────────────────────────────


class CLIPOracle:
    """Thin wrapper around a frozen image-text dual encoder: any Hugging Face
    `transformers` CLIP/SigLIP checkpoint, or an OpenCLIP model given as
    "hf-hub:<repo>" (e.g. hf-hub:timm/PE-Core-L-14-336 for Perception Encoder).
    `preprocess` maps the training pipeline's ImageNet-normalised 224px batches
    to the encoder's input; `embed_dim` is the joint embedding size."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if model_name.startswith("hf-hub:"):
            self._init_open_clip(model_name)
            return
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model_type = self.model.config.model_type
        ip = self.processor.image_processor
        size = getattr(ip, "crop_size", None) or ip.size
        self.image_size = size.get("height") or size.get("shortest_edge")
        self.preprocess = RenormalizeForCLIP(
            ip.image_mean, ip.image_std, self.image_size
        ).to(self.device)
        self.embed_dim = self.encode_text(["a photo"]).shape[-1]

    def _init_open_clip(self, model_name: str):
        import open_clip  # optional backend, only needed for hf-hub: models

        self.model, _ = open_clip.create_model_from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_type = "open_clip"
        size = self.model.visual.image_size
        self.image_size = size[0] if isinstance(size, (tuple, list)) else size
        self.preprocess = RenormalizeForCLIP(
            self.model.visual.image_mean, self.model.visual.image_std, self.image_size
        ).to(self.device)
        self.embed_dim = self.encode_text(["a photo"]).shape[-1]

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """(N, D) unit-normalised text embeddings."""
        if self.model_type == "open_clip":
            feats = self.model.encode_text(self.tokenizer(texts).to(self.device))
            return F.normalize(feats.float(), dim=-1)
        # SigLIP was trained on text padded to its full length (64 tokens for
        # most checkpoints, 16 for so400m-patch14-224)
        pad = (
            dict(
                padding="max_length",
                max_length=self.model.config.text_config.max_position_embeddings,
            )
            if self.model_type.startswith("siglip")
            else dict(padding=True)
        )
        inputs = self.processor(
            text=texts, return_tensors="pt", truncation=True, **pad
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, D) unit-normalised image embeddings from CLIP-preprocessed pixels."""
        if self.model_type == "open_clip":
            return F.normalize(self.model.encode_image(pixel_values).float(), dim=-1)
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


def compute_multi_concept_subspace(
    oracle: CLIPOracle,
    prompt_pairs: List[Tuple[List[str], List[str]]],
    k: int = 3,
) -> torch.Tensor:
    """Concept subspace for several attributes: each attribute's top-k text
    subspace (compute_concept_subspace), stacked and re-orthonormalised.
    One attribute returns exactly compute_concept_subspace's result."""
    parts = [
        compute_concept_subspace(oracle, pos, neg, k=k) for pos, neg in prompt_pairs
    ]
    if len(parts) == 1:
        return parts[0]
    q, _ = torch.linalg.qr(torch.cat(parts).T)  # (D, sum k) orthonormal columns
    return q.T.contiguous()


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


# ── Whitened (LEACE-style) erasure ────────────────────────────────────────────
#
# Orthogonal projection removes the component of V_I along the text directions,
# but CLIP image embeddings are strongly anisotropic, so the concept stays
# linearly decodable from correlated directions (probe_clip_target.py: place /
# gender probes are unchanged by it).  The whitened eraser removes everything
# in V_I that linearly covaries with the text-subspace scores, using only the
# unlabeled image covariance — still label-free.


class Eraser(NamedTuple):
    mu: torch.Tensor  # (D,)   train-set mean of V_I
    M: torch.Tensor  # (D, D) erasure map:  x' = x − (x − μ) Mᵀ
    R: (
        torch.Tensor
    )  # (r, D) whitened concept scores:  z = (x − μ) Rᵀ  (unit variance, decorrelated)

    def to(self, device) -> "Eraser":
        return Eraser(*(t.to(device) for t in self))


def fit_leace(x: torch.Tensor, cross_cov: torch.Tensor, eps: float = 1e-6) -> Eraser:
    """LEACE-style eraser (Belrose et al. 2023) for features x (N, D).

    Erases the span of `cross_cov` (D, r) — the covariance between x and the
    variable(s) to erase — in whitened coordinates:
        x' = x − (x − μ) Mᵀ,   M = W⁺ Q Qᵀ W,   W = Σ^{-1/2},
    with Q an orthonormal basis of W·cross_cov.  Afterwards no linear function
    of x' covaries with those variables.  Also returns R = Qᵀ W, which maps x
    to the whitened scores of what the eraser removes.  Fit in float64 on CPU.
    """
    x = x.detach().double().cpu()
    mu = x.mean(0)
    evals, evecs = torch.linalg.eigh(torch.cov((x - mu).T))
    evals = evals.clamp_min(eps)
    W = (evecs / evals.sqrt()) @ evecs.T
    W_pinv = (evecs * evals.sqrt()) @ evecs.T
    Q, _ = torch.linalg.qr(W @ cross_cov.detach().double().cpu())
    return Eraser(mu.float(), (W_pinv @ Q @ Q.T @ W).float(), (Q.T @ W).float())


def fit_leace_eraser(v_i: torch.Tensor, subspace: torch.Tensor) -> Eraser:
    """Label-free eraser for the text subspace: cross-covariance of V_I with its
    own subspace scores (V_I · b_k) is Σ Sᵀ."""
    x = v_i.detach().double().cpu()
    cross_cov = torch.cov((x - x.mean(0)).T) @ subspace.detach().double().cpu().T
    return fit_leace(x, cross_cov)


def leace_project(v_i: torch.Tensor, eraser: Eraser) -> torch.Tensor:
    return v_i - (v_i - eraser.mu) @ eraser.M.T


def whitened_concept_scores(v_i: torch.Tensor, eraser: Eraser) -> torch.Tensor:
    """(B, r) label-free concept scores of V_I: its text-subspace scores,
    centred and whitened with the train-set image covariance."""
    return (v_i - eraser.mu) @ eraser.R.T


@torch.no_grad()
def embed_dataset(
    oracle: CLIPOracle, dataset: Dataset, batch_size: int = 256, num_workers: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One ordered CLIP pass over a (image, label, concept) dataset whose images
    are ImageNet-normalised (as in training).  Returns CPU tensors
    v_i (N, D), labels (N,), concepts (N,), aligned with dataset indices."""
    renorm = oracle.preprocess
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    v_i, labels, concepts = [], [], []
    for images, y, c in loader:
        v_i.append(oracle.encode_images(renorm(images.to(oracle.device))).float().cpu())
        labels.append(y)
        concepts.append(c)
    return torch.cat(v_i), torch.cat(labels), torch.cat(concepts)


@torch.no_grad()
def zeroshot_concept_labels(
    oracle: CLIPOracle,
    prompts_pos: List[str],
    prompts_neg: List[str],
    v_i: torch.Tensor,
) -> torch.Tensor:
    """CLIP zero-shot concept prediction: 1 if V_I is closer to the mean positive
    prompt embedding than to the mean negative one."""
    d = oracle.encode_text(prompts_pos).mean(0) - oracle.encode_text(prompts_neg).mean(
        0
    )
    return (v_i.to(d.device) @ d > 0).long().cpu()


@torch.no_grad()
def zeroshot_concept_codes(
    oracle: CLIPOracle,
    prompt_pairs: List[Tuple[List[str], List[str]]],
    v_i: torch.Tensor,
) -> torch.Tensor:
    """Joint zero-shot concept code sum_j 2^j * c_j over several attributes
    (attribute j = prompt_pairs[j]); one attribute gives the plain 0/1 label."""
    code = torch.zeros(len(v_i), dtype=torch.long)
    for j, (pos, neg) in enumerate(prompt_pairs):
        code += zeroshot_concept_labels(oracle, pos, neg, v_i) << j
    return code


# ── Distillation target computation ──────────────────────────────────────────


@torch.no_grad()
def compute_distillation_targets(
    oracle: CLIPOracle,
    pixel_values: torch.Tensor,
    subspace: torch.Tensor,
    renormalize: bool = True,
    eraser: Eraser | None = None,
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
    eraser       : Eraser from fit_leace_eraser — if given, V_I_perp uses the
                   whitened eraser instead of orthogonal projection of `subspace`

    Returns
    -------
    v_i      : (B, D)  original CLIP image embeddings (proj head target)
    v_i_perp : (B, D)  concept-scrubbed embeddings    (backbone target)
    """
    v_i = oracle.encode_images(pixel_values)  # (B, D)
    if eraser is not None:
        v_i_perp = leace_project(v_i, eraser)  # (B, D)
    else:
        v_i_perp = orthogonal_project_subspace(v_i, subspace)  # (B, D)
    if renormalize:
        v_i_perp = F.normalize(v_i_perp, dim=-1)
    return v_i, v_i_perp
