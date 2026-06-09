#!/usr/bin/env python
"""
waterbirds_train.py — Waterbirds debiasing pilot with WordNet-derived concept direction.

The "water background" concept direction is built automatically:
  pos: WordNet synonyms of "water"  (water-related lemmas)
  neg: randomly sampled non-water nouns from WordNet

Usage
-----
    python waterbirds_train.py
    python waterbirds_train.py --epochs 5 --batch_size 32
"""

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch

from clip_debias.clip_oracle import CLIPOracle, build_concept_direction
from clip_debias.config import DebiasingConfig
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model
from clip_debias.trainer import Trainer
from clip_debias.waterbirds_data import build_waterbirds_dataloaders
from clip_debias.wordnet_utils import get_random_base_nouns, get_synonyms


@dataclass
class WaterbirdsConfig(DebiasingConfig):
    concept_attr: str = "place"               # background label — used for probe eval
    checkpoint_dir: str = "./checkpoints_waterbirds"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(cfg: WaterbirdsConfig) -> WaterbirdsConfig:
    parser = argparse.ArgumentParser(description="Waterbirds CLIP debiasing pilot")
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
    cfg = parse_args(WaterbirdsConfig())
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_waterbirds_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}  "
          f"Test: {len(test_loader.dataset):,}")

    # ── CLIP oracle ───────────────────────────────────────────────────────────
    print(f"\nLoading CLIP ({cfg.clip_model}) …")
    oracle = CLIPOracle(cfg.clip_model, device=device)

    # ── WordNet concept words ─────────────────────────────────────────────────
    # filter_by_name=True (default) keeps only synsets whose name contains
    # "water", excluding the urine.n.01 synset that also lists "water" as lemma.
    pos_words = get_synonyms("water")
    neg_words = get_random_base_nouns(
        n=len(pos_words) * 5,
        exclude_words=set(pos_words),
        seed=cfg.seed,
    )

    print(f"\nConcept direction words:")
    print(f"  pos ({len(pos_words)}): {pos_words}")
    print(f"  neg ({len(neg_words)}): {neg_words[:10]} ...")

    # ── Concept direction ─────────────────────────────────────────────────────
    print("\nComputing concept direction …")
    v_t_hat = build_concept_direction(
        oracle,
        prompts_pos=pos_words,
        prompts_neg=neg_words,
    )
    print(f"  ‖v_t_hat‖ = {v_t_hat.norm().item():.4f}  (should be 1.0)")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {cfg.backbone} classifier …")
    model = build_model(cfg, num_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Baseline evaluation ───────────────────────────────────────────────────
    print("\n--- Baseline evaluation (before debiasing) ---")
    run_evaluation(model, train_loader, test_loader, device, cfg.amp,
                   label="before debiasing")

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
    run_evaluation(model, train_loader, test_loader, device, cfg.amp,
                   label="after debiasing")


if __name__ == "__main__":
    main()
