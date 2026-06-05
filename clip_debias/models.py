"""
models.py — All neural network components.

Components
----------
ProjectionHead   : small MLP mapping E(x) → CLIP embedding space
DebiasedClassifier : backbone + classifier head + projection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── Projection Head ───────────────────────────────────────────────────────────


class ProjectionHead(nn.Module):
    """
    2-layer MLP: backbone_dim → hidden_dim → out_dim (CLIP space).
    BatchNorm + GELU in the hidden layer; L2-normalize the output so it
    lives on the unit hypersphere alongside CLIP embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit-norm for cosine arithmetic


# ── Backbone helpers ──────────────────────────────────────────────────────────


def _build_resnet50(num_classes: int):
    """ResNet-50 with the final FC replaced for num_classes."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    embed_dim = backbone.fc.in_features  # 2048
    backbone.fc = nn.Linear(embed_dim, num_classes)
    return backbone, embed_dim


def _build_vit_b16(num_classes: int):
    """ViT-B/16 with the head replaced for num_classes."""
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    embed_dim = backbone.heads.head.in_features  # 768
    backbone.heads.head = nn.Linear(embed_dim, num_classes)
    return backbone, embed_dim


# ── Main model ────────────────────────────────────────────────────────────────


class DebiasedClassifier(nn.Module):
    """
    Wraps a backbone classifier and attaches a projection head.

    Forward pass returns a dict so callers can access everything they need
    without multiple forward passes:

        {
          "logits"  : (B, num_classes)  — task classification logits
          "embed"   : (B, embed_dim)    — penultimate activations E(x)
          "proj"    : (B, clip_dim)     — P(E(x)), unit-normalised
        }

    The hook-based approach extracts E(x) (penultimate layer) without
    modifying the backbone's forward signature.
    """

    def __init__(self, backbone_name: str, num_classes: int, cfg):
        super().__init__()

        if backbone_name == "resnet50":
            self.backbone, embed_dim = _build_resnet50(num_classes)
            # Hook target: avgpool output  (B, 2048, 1, 1) → squeezed to (B, 2048)
            self._hook_layer = self.backbone.avgpool
        elif backbone_name == "vit_b_16":
            self.backbone, embed_dim = _build_vit_b16(num_classes)
            # Hook target: the layer before the classification head
            self._hook_layer = self.backbone.encoder
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Choose 'resnet50' or 'vit_b_16'."
            )

        self.proj_head = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=cfg.proj_hidden_dim,
            out_dim=cfg.proj_out_dim,
        )

        self._embed: torch.Tensor | None = None
        self._hook_handle = self._hook_layer.register_forward_hook(self._save_embed)

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _save_embed(self, module, input, output):
        """Forward hook: store penultimate activations."""
        if isinstance(output, torch.Tensor):
            # ResNet: avgpool output is (B, C, 1, 1) — flatten to (B, C)
            self._embed = output.flatten(1)
        else:
            # ViT encoder returns a named tuple; CLS token is index 0
            self._embed = output.last_hidden_state[:, 0]

    def remove_hooks(self):
        """Call this if the model is no longer needed to free resources."""
        self._hook_handle.remove()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.backbone(x)  # triggers the hook
        embed = self._embed  # (B, embed_dim)
        proj = self.proj_head(embed)  # (B, clip_dim), unit-normalised
        return {"logits": logits, "embed": embed, "proj": proj}


# ── Convenience factory ───────────────────────────────────────────────────────


def build_model(cfg, num_classes: int = 2) -> DebiasedClassifier:
    return DebiasedClassifier(
        backbone_name=cfg.backbone,
        num_classes=num_classes,
        cfg=cfg,
    )
