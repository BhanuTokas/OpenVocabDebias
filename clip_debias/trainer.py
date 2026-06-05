"""
trainer.py — Training loop with AMP, checkpointing, and ERM-mode support.

The same Trainer class handles both ERM (lambda_align=0, lambda_repulse=0)
and the full debiasing setup.  When both debiasing weights are zero, CLIP
is never called, saving memory and time for the baseline run.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .clip_oracle import CLIPOracle, compute_distillation_targets
from .clip_preprocess import RenormalizeForCLIP
from .losses import DebiasingLoss
from .models import DebiasedClassifier


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _fmt(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


class Trainer:
    """
    Unified trainer for ERM baseline and debiasing runs.

    ERM mode
    --------
    Set lambda_align=0 and lambda_repulse=0 in cfg (use erm_config()).
    In this case oracle and v_t_hat may be passed as None — CLIP is
    never called, keeping memory usage and wall time identical to a
    plain fine-tuning run.

    Debiasing mode
    --------------
    Pass a CLIPOracle and a pre-computed unit concept direction v_t_hat.
    The combined loss (task + align + repulse) is used.

    Checkpoints
    -----------
    Saved to <checkpoint_dir>/<run_name>/epoch_NN.pt.
    Best val-acc model → <checkpoint_dir>/<run_name>/best.pt.
    """

    def __init__(
        self,
        model: DebiasedClassifier,
        cfg,
        oracle: CLIPOracle | None = None,
        v_t_hat: torch.Tensor | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self._erm_mode = cfg.lambda_align == 0.0 and cfg.lambda_repulse == 0.0
        if not self._erm_mode and (oracle is None or v_t_hat is None):
            raise ValueError(
                "oracle and v_t_hat are required when lambda_align or "
                "lambda_repulse > 0.  Pass oracle=None only for ERM mode."
            )

        self.oracle = oracle
        self.v_t_hat = v_t_hat.to(self.device) if v_t_hat is not None else None
        self.renorm = (
            RenormalizeForCLIP().to(self.device) if not self._erm_mode else None
        )

        self.criterion = DebiasingLoss(cfg)
        self.scaler = GradScaler(enabled=cfg.amp)
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler: OneCycleLR | None = None

        # Per-run checkpoint subdirectory so runs don't overwrite each other
        self.ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self._best_val_acc = 0.0

    def _setup_scheduler(self, steps_per_epoch: int):
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.lr,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.cfg.warmup_epochs / self.cfg.epochs,
        )

    # ── Single training step ──────────────────────────────────────────────────

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Distillation targets — skipped entirely in ERM mode
        v_i_perp = None
        if not self._erm_mode:
            clip_images = self.renorm(images)
            v_i_perp = compute_distillation_targets(
                self.oracle, clip_images, self.v_t_hat, renormalize=True
            )

        with autocast(enabled=self.cfg.amp):
            out = self.model(images)
            loss, info = self.criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,  # None is safe when weights are 0
                v_t_hat=self.v_t_hat,  # None is safe when weights are 0
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        return info, _accuracy(out["logits"], labels)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_correct = total = 0
        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                loss = nn.functional.cross_entropy(out["logits"], labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (out["logits"].argmax(1) == labels).sum().item()
            total += images.size(0)
        self.model.train()
        return {"val_loss": total_loss / total, "val_acc": total_correct / total}

    # ── Full training run ─────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader) -> str:
        """
        Train for cfg.epochs epochs.

        Returns
        -------
        best_ckpt_path : str — path to the best checkpoint (by val acc)
        """
        self._setup_scheduler(len(train_loader))
        self.model.to(self.device).train()

        mode_str = (
            "ERM"
            if self._erm_mode
            else (
                f"debias  λ_align={self.cfg.lambda_align}  "
                f"λ_repulse={self.cfg.lambda_repulse}"
            )
        )
        print(f"\n[{self.cfg.run_name}]  seed={self.cfg.seed}  {mode_str}")

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            running = {
                k: 0.0
                for k in [
                    "loss_task",
                    "loss_align",
                    "loss_repulse",
                    "loss_total",
                    "acc",
                ]
            }
            n = 0

            for step, (images, labels, _) in enumerate(train_loader, 1):
                info, acc = self._train_step(images, labels)
                for k in ["loss_task", "loss_align", "loss_repulse", "loss_total"]:
                    running[k] += info[k]
                running["acc"] += acc
                n += 1

                if step % self.cfg.log_interval == 0:
                    avg = {k: v / n for k, v in running.items()}
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"  [ep {epoch}/{self.cfg.epochs}  step {step}/{len(train_loader)}]"
                        f"  lr={lr:.2e}  " + _fmt(avg)
                    )

            val = self._validate(val_loader)
            avg = {k: v / n for k, v in running.items()}
            print(f"\n  === Epoch {epoch} ({time.time()-t0:.1f}s) ===")
            print(f"    train — " + _fmt(avg))
            print(f"    val   — " + _fmt(val))

            # ── Checkpoint ───────────────────────────────────────────────────
            ckpt = os.path.join(self.ckpt_dir, f"epoch_{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_acc": val["val_acc"],
                    "run_name": self.cfg.run_name,
                    "seed": self.cfg.seed,
                },
                ckpt,
            )

            if val["val_acc"] > self._best_val_acc:
                self._best_val_acc = val["val_acc"]
                best = os.path.join(self.ckpt_dir, "best.pt")
                torch.save(self.model.state_dict(), best)
                print(f"    ✓ best val_acc={val['val_acc']:.4f}  → {best}")

        print()
        return os.path.join(self.ckpt_dir, "best.pt")
