"""
trainer.py — Full training loop with AMP, checkpointing, and logging.
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

from .clip_oracle import CLIPOracle, build_concept_direction, compute_distillation_targets
from .clip_preprocess import RenormalizeForCLIP
from .losses import DebiasingLoss
from .models import DebiasedClassifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _format_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the debiasing training loop.

    Parameters
    ----------
    model     : DebiasedClassifier
    oracle    : CLIPOracle (frozen)
    v_t_hat   : (D,) unit concept direction, pre-computed
    cfg       : DebiasingConfig
    """

    def __init__(
        self,
        model: DebiasedClassifier,
        oracle: CLIPOracle,
        v_t_hat: torch.Tensor,
        cfg,
    ):
        self.model = model
        self.oracle = oracle
        self.v_t_hat = v_t_hat.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.criterion = DebiasingLoss(cfg)
        self.renorm = RenormalizeForCLIP().to(self.device)
        self.scaler = GradScaler(enabled=cfg.amp)

        # Only parameters in the backbone + projection head are trained.
        # CLIP is frozen inside CLIPOracle.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler is set up after we know the number of steps per epoch.
        self.scheduler: OneCycleLR | None = None

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._best_val_acc = 0.0

    def _setup_scheduler(self, steps_per_epoch: int):
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.lr,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.cfg.warmup_epochs / self.cfg.epochs,
        )

    # ── Single step ───────────────────────────────────────────────────────────

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Compute distillation targets with frozen CLIP (no grad)
        clip_images = self.renorm(images)
        v_i_perp = compute_distillation_targets(
            self.oracle, clip_images, self.v_t_hat, renormalize=True
        )  # (B, D)

        with autocast(enabled=self.cfg.amp):
            out = self.model(images)
            loss, info = self.criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,
                v_t_hat=self.v_t_hat,
            )

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        acc = _accuracy(out["logits"], labels)
        return info, acc

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with autocast(enabled=self.cfg.amp):
                out = self.model(images)
                loss = torch.nn.functional.cross_entropy(out["logits"], labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (out["logits"].argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        self.model.train()
        return {
            "val_loss": total_loss / total_samples,
            "val_acc": total_correct / total_samples,
        }

    # ── Full training run ─────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader):
        self._setup_scheduler(len(train_loader))
        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.time()
            running = {
                "loss_task": 0.0,
                "loss_align": 0.0,
                "loss_repulse": 0.0,
                "loss_total": 0.0,
                "acc": 0.0,
            }
            n_steps = 0

            for step, (images, labels, _concept) in enumerate(train_loader, 1):
                info, acc = self._train_step(images, labels)
                for k in ["loss_task", "loss_align", "loss_repulse", "loss_total"]:
                    running[k] += info[k]
                running["acc"] += acc
                n_steps += 1

                if step % self.cfg.log_interval == 0:
                    avg = {k: v / n_steps for k, v in running.items()}
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"[Epoch {epoch}/{self.cfg.epochs}  step {step}/{len(train_loader)}]  "
                        f"lr={lr:.2e}  " + _format_metrics(avg)
                    )

            # ── End of epoch ─────────────────────────────────────────────────
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - epoch_start
            avg = {k: v / n_steps for k, v in running.items()}
            print(
                f"\n=== Epoch {epoch} done ({elapsed:.1f}s) ===\n"
                f"  Train — " + _format_metrics(avg) + "\n"
                f"  Val   — " + _format_metrics(val_metrics) + "\n"
            )

            # ── Checkpoint ───────────────────────────────────────────────────
            val_acc = val_metrics["val_acc"]
            ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch:02d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ New best val_acc={val_acc:.4f}  saved to {best_path}\n")
