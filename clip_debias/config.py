from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Per-attribute concept prompt library ─────────────────────────────────────
# Maps CelebA attribute name → (positive_prompts, negative_prompts).
# Positive = the attribute is present; negative = the attribute is absent.
CONCEPT_PROMPT_LIBRARY: Dict[str, Tuple[List[str], List[str]]] = {
    "Male": (
        ["a photo of a man", "a photo of a male person", "male", "a man"],
        ["a photo of a woman", "a photo of a female person", "female", "a woman"],
    ),
    "Young": (
        [
            "a photo of a young person",
            "a young adult",
            "young",
            "a photo of a young face",
        ],
        [
            "a photo of an old person",
            "an elderly person",
            "old",
            "a photo of an aged face",
        ],
    ),
    "Smiling": (
        [
            "a photo of a smiling person",
            "a person with a smile",
            "smiling",
            "a happy face",
        ],
        [
            "a photo of a person not smiling",
            "a serious face",
            "not smiling",
            "a neutral expression",
        ],
    ),
    "Eyeglasses": (
        [
            "a photo of a person wearing glasses",
            "wearing eyeglasses",
            "glasses on face",
            "a person with spectacles",
        ],
        [
            "a photo of a person without glasses",
            "no glasses",
            "a person without eyeglasses",
            "bare face without glasses",
        ],
    ),
    "Bald": (
        ["a photo of a bald person", "a bald head", "bald", "a person with no hair"],
        [
            "a photo of a person with hair",
            "hair on head",
            "not bald",
            "a person with a full head of hair",
        ],
    ),
    "Blond_Hair": (
        [
            "a photo of a person with blond hair",
            "blond hair",
            "blonde",
            "a person with light golden hair",
        ],
        [
            "a photo of a person without blond hair",
            "dark hair",
            "non-blond hair",
            "a person with brown or black hair",
        ],
    ),
    "Heavy_Makeup": (
        [
            "a photo of a person with heavy makeup",
            "wearing a lot of makeup",
            "heavy cosmetics",
            "full face of makeup",
        ],
        [
            "a photo of a person without makeup",
            "no makeup",
            "a natural face",
            "minimal cosmetics",
        ],
    ),
    "Pale_Skin": (
        [
            "a photo of a person with pale skin",
            "very light skin",
            "pale complexion",
            "fair skin",
        ],
        [
            "a photo of a person with darker skin",
            "darker complexion",
            "non-pale skin",
            "olive or dark skin",
        ],
    ),
    "Chubby": (
        [
            "a photo of a chubby person",
            "overweight face",
            "chubby cheeks",
            "a heavier person",
        ],
        ["a photo of a thin person", "slim face", "a slender person", "not chubby"],
    ),
    "Wearing_Hat": (
        [
            "a photo of a person wearing a hat",
            "hat on head",
            "wearing headwear",
            "a person with a hat",
        ],
        [
            "a photo of a person without a hat",
            "no hat",
            "bareheaded person",
            "a person with no headwear",
        ],
    ),
    "Attractive": (
        [
            "a photo of an attractive person",
            "a beautiful person",
            "a good-looking person",
            "attractive face",
        ],
        [
            "a photo of an unattractive person",
            "a plain-looking person",
            "not attractive",
            "ordinary face",
        ],
    ),
    "No_Beard": (
        [
            "a photo of a person with no beard",
            "clean-shaven face",
            "no facial hair",
            "a beardless person",
        ],
        [
            "a photo of a person with a beard",
            "beard on face",
            "facial hair",
            "a bearded person",
        ],
    ),
    "Mustache": (
        [
            "a photo of a person with a mustache",
            "mustache on upper lip",
            "facial hair above mouth",
            "a mustachioed person",
        ],
        [
            "a photo of a person without a mustache",
            "no mustache",
            "clean upper lip",
            "a person with no facial hair above lip",
        ],
    ),
    "Goatee": (
        [
            "a photo of a person with a goatee",
            "goatee beard",
            "chin beard",
            "a person with a goatee",
        ],
        [
            "a photo of a person without a goatee",
            "no goatee",
            "clean chin",
            "a person with no chin beard",
        ],
    ),
    "Wearing_Lipstick": (
        [
            "a photo of a person wearing lipstick",
            "red lips",
            "lipstick on lips",
            "colored lip makeup",
        ],
        [
            "a photo of a person without lipstick",
            "no lipstick",
            "bare lips",
            "natural lip color",
        ],
    ),
    "Wearing_Earrings": (
        [
            "a photo of a person wearing earrings",
            "earrings on ears",
            "ear jewelry",
            "a person with earrings",
        ],
        [
            "a photo of a person without earrings",
            "no earrings",
            "bare ears",
            "a person with no ear jewelry",
        ],
    ),
}


def get_concept_prompts(attr: str) -> Tuple[List[str], List[str]]:
    """Return (pos_prompts, neg_prompts) for a CelebA attribute name.

    Falls back to generic present/absent phrasing for unlisted attributes.
    """
    if attr in CONCEPT_PROMPT_LIBRARY:
        return CONCEPT_PROMPT_LIBRARY[attr]
    label = attr.replace("_", " ").lower()
    return (
        [f"a photo of a person with {label}", f"showing {label}", label],
        [f"a photo of a person without {label}", f"no {label}", f"not {label}"],
    )


@dataclass
class DebiasingConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    celeba_root: str = "./data/celeba"
    target_attr: str = "Attractive"  # classifier target (CelebA attribute)
    concept_attr: str = "Male"  # concept to debias / probe
    batch_size: int = 64
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "resnet50"  # "resnet50" | "vit_b_16"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512

    # Projection head (E(x) → CLIP space)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512  # must match clip_embed_dim

    # ── Concept prompts ───────────────────────────────────────────────────────
    # Defaults are populated from CONCEPT_PROMPT_LIBRARY at config-factory time.
    # Override by passing concept_prompts_pos / concept_prompts_neg explicitly.
    concept_prompts_pos: List[str] = field(
        default_factory=lambda: CONCEPT_PROMPT_LIBRARY["Male"][0]
    )
    concept_prompts_neg: List[str] = field(
        default_factory=lambda: CONCEPT_PROMPT_LIBRARY["Male"][1]
    )

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 10
    lr: float = 1e-4  # backbone learning rate
    lr_proj: float = 1e-3  # proj head learning rate (simpler task → higher LR)
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    # Backbone loss weights (set lambda_align=0, lambda_repulse=0 for ERM baseline)
    lambda_task: float = 1.0
    lambda_align: float = 1.0
    lambda_repulse: float = 0.5
    lambda_task_warmup: bool = True            # ramp lambda_task up from calibrated init
    lambda_task_warmup_schedule: str = "cosine"  # "linear" | "cosine"

    # ── Run identity ──────────────────────────────────────────────────────────
    run_name: str = "debias"
    seed: int = 42

    # ── Misc ──────────────────────────────────────────────────────────────────
    device: str = "cuda"
    amp: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50

    # ── Evaluation ────────────────────────────────────────────────────────────
    probe_max_iter: int = 1000


# ── Pre-defined ablation configurations ──────────────────────────────────────


def _apply_overrides(cfg: DebiasingConfig, overrides: dict) -> DebiasingConfig:
    """Apply overrides and re-sync concept prompts if concept_attr changed."""
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # If concept_attr was overridden but prompts were not, auto-populate from library.
    if (
        "concept_attr" in overrides
        and "concept_prompts_pos" not in overrides
        and "concept_prompts_neg" not in overrides
    ):
        pos, neg = get_concept_prompts(cfg.concept_attr)
        cfg.concept_prompts_pos = pos
        cfg.concept_prompts_neg = neg
    return cfg


def erm_config(**overrides) -> DebiasingConfig:
    """Pure ERM baseline — no debiasing losses, no proj head optimizer."""
    cfg = DebiasingConfig(
        run_name="erm",
        lambda_task=1.0,
        lambda_align=0.0,
        lambda_repulse=0.0,
    )
    return _apply_overrides(cfg, overrides)


def align_only_config(**overrides) -> DebiasingConfig:
    """Ablation: backbone alignment loss only, no repulsion."""
    cfg = DebiasingConfig(
        run_name="align_only",
        lambda_task=1.0,
        lambda_align=1.0,
        lambda_repulse=0.0,
    )
    return _apply_overrides(cfg, overrides)


def repulse_only_config(**overrides) -> DebiasingConfig:
    """Ablation: repulsion only, no alignment."""
    cfg = DebiasingConfig(
        run_name="repulse_only",
        lambda_task=1.0,
        lambda_align=0.0,
        lambda_repulse=0.5,
    )
    return _apply_overrides(cfg, overrides)


def full_config(**overrides) -> DebiasingConfig:
    """Full method: task + align + repulse on backbone, reconstruction on proj head."""
    cfg = DebiasingConfig(
        run_name="full",
        lambda_task=1.0,
        lambda_align=1.0,
        lambda_repulse=0.5,
    )
    return _apply_overrides(cfg, overrides)


def full_strong_config(**overrides) -> DebiasingConfig:
    """Full method with stronger debiasing signal."""
    cfg = DebiasingConfig(
        run_name="full_strong",
        lambda_task=1.0,
        lambda_align=2.0,
        lambda_repulse=1.0,
    )
    return _apply_overrides(cfg, overrides)
