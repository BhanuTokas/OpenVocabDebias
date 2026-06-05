from dataclasses import dataclass, field
from typing import List


@dataclass
class DebiasingConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    celeba_root: str = "./data/celeba"
    target_attr: str = "Attractive"       # classifier target (CelebA attribute)
    concept_attr: str = "Male"            # concept to debias / probe
    batch_size: int = 64
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "resnet50"            # "resnet50" | "vit_b_16"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512

    # Projection head (E(x) → CLIP space)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512              # must match clip_embed_dim

    # ── Concept prompts ───────────────────────────────────────────────────────
    concept_prompts_pos: List[str] = field(default_factory=lambda: [
        "a photo of a man", "a photo of a male person", "male", "a man",
    ])
    concept_prompts_neg: List[str] = field(default_factory=lambda: [
        "a photo of a woman", "a photo of a female person", "female", "a woman",
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    # Loss weights (set lambda_align=0, lambda_repulse=0 for ERM baseline)
    lambda_task: float = 1.0
    lambda_align: float = 1.0
    lambda_repulse: float = 0.5

    # ── Run identity ──────────────────────────────────────────────────────────
    # Set automatically by the training script; used for checkpoint naming.
    run_name: str = "debias"             # e.g. "erm", "align_only", "full"
    seed: int = 42

    # ── Misc ──────────────────────────────────────────────────────────────────
    device: str = "cuda"
    amp: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50

    # ── Evaluation ────────────────────────────────────────────────────────────
    probe_max_iter: int = 1000


# ── Pre-defined ablation configurations ──────────────────────────────────────

def erm_config(**overrides) -> DebiasingConfig:
    """Pure ERM baseline — no debiasing losses."""
    cfg = DebiasingConfig(
        run_name="erm",
        lambda_task=1.0,
        lambda_align=0.0,
        lambda_repulse=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def align_only_config(**overrides) -> DebiasingConfig:
    """Ablation: alignment loss only, no repulsion."""
    cfg = DebiasingConfig(
        run_name="align_only",
        lambda_task=1.0,
        lambda_align=1.0,
        lambda_repulse=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def repulse_only_config(**overrides) -> DebiasingConfig:
    """Ablation: repulsion loss only, no alignment."""
    cfg = DebiasingConfig(
        run_name="repulse_only",
        lambda_task=1.0,
        lambda_align=0.0,
        lambda_repulse=0.5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def full_config(**overrides) -> DebiasingConfig:
    """Full method: task + align + repulse."""
    cfg = DebiasingConfig(
        run_name="full",
        lambda_task=1.0,
        lambda_align=1.0,
        lambda_repulse=0.5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def full_strong_config(**overrides) -> DebiasingConfig:
    """Full method with stronger debiasing signal."""
    cfg = DebiasingConfig(
        run_name="full_strong",
        lambda_task=1.0,
        lambda_align=2.0,
        lambda_repulse=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg