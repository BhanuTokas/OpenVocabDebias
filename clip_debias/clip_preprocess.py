"""
clip_preprocess.py — Re-normalise ImageNet-normalised tensors for CLIP.

The backbone receives images normalised with ImageNet stats.
CLIP expects its own normalisation.  Rather than loading images twice,
we undo ImageNet normalisation and apply CLIP normalisation on-the-fly,
keeping the data pipeline single-pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# Values used by openai/clip-vit-base-patch32
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class RenormalizeForCLIP(nn.Module):
    """
    Converts a batch of ImageNet-normalised tensors (B, 3, H, W) into the
    encoder's normalisation (OpenAI CLIP's by default; SigLIP etc. pass their
    own mean/std), resizing to `size` if the encoder expects another input size.

    Registered as buffers so they automatically move with .to(device).
    """

    def __init__(self, mean=None, std=None, size: int | None = None):
        super().__init__()
        mean = _CLIP_MEAN if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = _CLIP_STD if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.size = size
        # shape: (1, 3, 1, 1) for broadcast over (B, C, H, W)
        self.register_buffer("in_mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("in_std", _IMAGENET_STD.view(1, 3, 1, 1))
        self.register_buffer("clip_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("clip_std", std.view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Undo ImageNet normalisation → [0, 1] range
        x = x * self.in_std + self.in_mean
        x = x.clamp(0.0, 1.0)
        if self.size is not None and tuple(x.shape[-2:]) != (self.size, self.size):
            x = F.interpolate(
                x, size=(self.size, self.size), mode="bicubic", antialias=True
            ).clamp(0.0, 1.0)
        # 2. Apply CLIP normalisation
        x = (x - self.clip_mean) / self.clip_std
        return x
