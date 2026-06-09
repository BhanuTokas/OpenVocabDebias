#!/usr/bin/env python
"""
waterbirds_quick_cpu_pilot.py — Quick CPU pilot for the Waterbirds pipeline.

Takes a small subset (640 train / 160 val / 160 test) and runs 5 training
epochs to verify every component connects correctly end-to-end.
"""

import torch
from torch.utils.data import DataLoader, Subset

from clip_debias.clip_oracle import CLIPOracle, build_concept_direction
from clip_debias.config import DebiasingConfig
from clip_debias.data import get_eval_transform, get_train_transform
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.trainer import Trainer
from clip_debias.waterbirds_data import WaterbirdsDebiasDataset
from clip_debias.wordnet_utils import get_random_base_nouns, get_synonyms
from dataclasses import dataclass


@dataclass
class TestConfig(DebiasingConfig):
    concept_attr: str = "place"
    batch_size: int = 8
    num_workers: int = 0       # required on CPU / MPS for stability
    epochs: int = 5
    amp: bool = False          # AMP needs CUDA
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints_test"
    log_interval: int = 1


def tiny_loader(split: str, transform, n: int, cfg: TestConfig, shuffle: bool) -> DataLoader:
    ds = WaterbirdsDebiasDataset(split, transform=transform)
    sub = Subset(ds, list(range(min(n, len(ds)))))
    return DataLoader(sub, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=False)


def main():
    cfg = TestConfig()
    print("=" * 60)
    print("  Waterbirds quick CPU pilot  (640 train / 160 val / 160 test)")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────
    print("\n[1] Loading dataset subsets …")
    train_loader = tiny_loader("train",      get_train_transform(), 640, cfg, shuffle=True)
    val_loader   = tiny_loader("validation", get_eval_transform(),  160, cfg, shuffle=False)
    test_loader  = tiny_loader("test",       get_eval_transform(),  160, cfg, shuffle=False)

    images, labels, places = next(iter(train_loader))
    print(f"    batch shape : {images.shape}")
    print(f"    labels      : {labels.tolist()}")
    print(f"    places      : {places.tolist()}")

    # ── CLIP oracle ───────────────────────────────────────────────
    print("\n[2] Loading CLIP oracle …")
    oracle = CLIPOracle(cfg.clip_model, device="cpu")

    # ── Concept direction ─────────────────────────────────────────
    print("\n[3] Building concept direction from WordNet …")
    pos_words = get_synonyms("water")
    neg_words = get_random_base_nouns(n=len(pos_words) * 5, exclude_words=set(pos_words), seed=42)
    print(f"    pos: {pos_words}")
    print(f"    neg (first 8): {neg_words[:8]}")

    v_t_hat = build_concept_direction(oracle, prompts_pos=pos_words, prompts_neg=neg_words)
    print(f"    v_t_hat shape={v_t_hat.shape}  norm={v_t_hat.norm().item():.4f}")
    assert abs(v_t_hat.norm().item() - 1.0) < 1e-4, "v_t_hat must be unit-normalised"

    # ── Model ─────────────────────────────────────────────────────
    print("\n[4] Building model …")
    model = build_model(cfg, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    trainable params: {n_params:,}")

    # ── Single forward pass check ─────────────────────────────────
    print("\n[5] Single forward pass …")
    model.eval()
    with torch.no_grad():
        out = model(images)
    print(f"    logits shape : {out['logits'].shape}")
    print(f"    embed  shape : {out['embed'].shape}")
    print(f"    proj   shape : {out['proj'].shape}")
    assert out["logits"].shape == (len(images), 2)
    assert out["proj"].shape[-1] == cfg.proj_out_dim

    # ── Baseline eval ─────────────────────────────────────────────
    print("\n[6] Baseline evaluation (before training) …")
    run_evaluation(model, train_loader, test_loader, device="cpu",
                   use_amp=False, label="before debiasing")

    # ── Training ──────────────────────────────────────────────────
    print(f"\n[7] Training for {cfg.epochs} epochs …")
    trainer = Trainer(model, oracle, v_t_hat, cfg)
    trainer.fit(train_loader, val_loader)

    # ── Post-training eval ────────────────────────────────────────
    print("\n[8] Post-training evaluation …")
    run_evaluation(model, train_loader, test_loader, device="cpu",
                   use_amp=False, label=f"after {cfg.epochs} epochs")

    print("\n" + "=" * 60)
    print("  All checks passed — pipeline is healthy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
