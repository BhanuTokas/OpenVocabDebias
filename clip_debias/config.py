from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DebiasingConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    celeba_root: str = "./data/celeba"          # root dir passed to torchvision CelebA
    target_attr: str = "Attractive"             # classifier target (CelebA attribute name)
    concept_attr: str = "Male"                  # concept to debias (used only for eval probing)
    batch_size: int = 64
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = "resnet50"                  # "resnet50" | "vit_b_16"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512

    # Projection head (E(x) → CLIP space)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512                     # must match clip_embed_dim

    # ── Concept prompts ───────────────────────────────────────────────────────
    # Prompt ensemble for gender concept direction
    concept_prompts: List[str] = field(default_factory=lambda: [
        "a photo of a man",
        "a photo of a woman",
        "a photo of a male person",
        "a photo of a female person",
        "male",
        "female",
        "a man",
        "a woman",
    ])
    # For a *directional* concept we split prompts into two poles and take
    # the difference.  Leave empty to just average all prompts (magnitude mode).
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

    # Loss weights  (total = λ_task·L_task + λ_align·L_align + λ_repulse·L_repulse)
    lambda_task: float = 1.0
    lambda_align: float = 1.0
    lambda_repulse: float = 0.5

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"                        # "cuda" | "cpu"
    amp: bool = True                            # automatic mixed precision
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50                      # steps between console log lines

    # ── Evaluation ────────────────────────────────────────────────────────────
    probe_epochs: int = 50
    probe_lr: float = 1e-3
