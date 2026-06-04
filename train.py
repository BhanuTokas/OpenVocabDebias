#!/usr/bin/env python
"""
train.py — Entry point for debiasing training.

Usage
-----
    python train.py
    python train.py --celeba_root /path/to/celeba --epochs 20 --batch_size 128
"""

import argparse
import random

import numpy as np
import torch

from clip_debias.clip_oracle import CLIPOracle, build_concept_direction
from clip_debias.config import DebiasingConfig
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(cfg: DebiasingConfig) -> DebiasingConfig:
    parser = argparse.ArgumentParser(description="CLIP-guided representation debiasing")
    parser.add_argument("--celeba_root", type=str, default=cfg.celeba_root)
    parser.add_argument("--target_attr", type=str, default=cfg.target_attr)
    parser.add_argument("--concept_attr", type=str, default=cfg.concept_attr)
    parser.add_argument("--backbone", type=str, default=cfg.backbone)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--lambda_task", type=float, default=cfg.lambda_task)
    parser.add_argument("--lambda_align", type=float, default=cfg.lambda_align)
    parser.add_argument("--lambda_repulse", type=float, default=cfg.lambda_repulse)
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.checkpoint_dir)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg.celeba_root = args.celeba_root
    cfg.target_attr = args.target_attr
    cfg.concept_attr = args.concept_attr
    cfg.backbone = args.backbone
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lambda_task = args.lambda_task
    cfg.lambda_align = args.lambda_align
    cfg.lambda_repulse = args.lambda_repulse
    cfg.checkpoint_dir = args.checkpoint_dir
    if args.no_amp:
        cfg.amp = False
    return cfg


def main():
    cfg = parse_args(DebiasingConfig())
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}  "
          f"Test: {len(test_loader.dataset):,}")

    # ── CLIP oracle ───────────────────────────────────────────────────────────
    print(f"\nLoading CLIP ({cfg.clip_model}) …")
    oracle = CLIPOracle(cfg.clip_model, device=device)

    print("Computing concept direction …")
    v_t_hat = build_concept_direction(
        oracle,
        prompts_pos=cfg.concept_prompts_pos,
        prompts_neg=cfg.concept_prompts_neg,
    )
    print(f"  ‖v_t_hat‖ = {v_t_hat.norm().item():.4f}  (should be 1.0)")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {cfg.backbone} classifier …")
    model = build_model(cfg, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Baseline evaluation (before debiasing) ────────────────────────────────
    print("\n--- Baseline evaluation (randomly-initialised classifier head) ---")
    run_evaluation(model, train_loader, test_loader, device, cfg.amp, label="before debiasing")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\nStarting debiasing training …")
    trainer = Trainer(model, oracle, v_t_hat, cfg)
    trainer.fit(train_loader, val_loader)

    # ── Load best checkpoint ──────────────────────────────────────────────────
    import os
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    print(f"\nLoading best checkpoint from {best_path} …")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ── Post-debiasing evaluation ─────────────────────────────────────────────
    run_evaluation(model, train_loader, test_loader, device, cfg.amp, label="after debiasing")


if __name__ == "__main__":
    main()
