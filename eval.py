#!/usr/bin/env python
"""
eval.py — Evaluate a saved checkpoint without re-training.

Usage
-----
    python eval.py --checkpoint checkpoints/best.pt
    python eval.py --checkpoint checkpoints/best.pt --split test
"""

import argparse

import torch

from clip_debias.config import DebiasingConfig
from clip_debias.data import build_dataloaders
from clip_debias.evaluate import run_evaluation
from clip_debias.models import build_model


def main():
    cfg = DebiasingConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--celeba_root", default=cfg.celeba_root)
    parser.add_argument("--backbone", default=cfg.backbone)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg.celeba_root = args.celeba_root
    cfg.backbone = args.backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    model = build_model(cfg, num_classes=2)
    state = torch.load(args.checkpoint, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    run_evaluation(model, train_loader, split_loader, device, cfg.amp, label=args.checkpoint)


if __name__ == "__main__":
    main()
