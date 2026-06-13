"""
trainer.py — Dual-optimizer training loop.

Two optimizers with separate responsibilities:

  optimizer_proj     — updates projection head only
                       loss: L_reconstruct = 1 - cos(P(E(x)), V_I)
                       goal: proj head becomes a faithful CLIP reconstructor

  optimizer_backbone — updates backbone only
                       loss: λ_task·L_task + λ_align·L_align + λ_repulse·L_repulse
                       goal: backbone classifies correctly while removing concept

In ERM mode (lambda_align=0, lambda_repulse=0) only optimizer_backbone is
used and CLIP is never called, keeping ERM a clean matched baseline.

Step order per batch
--------------------
1. Forward pass (single pass)
2. Backbone backward — uses out["proj"] from the original graph; proj_head.params
   accumulate contamination grads from L_align/L_repulse flowing through it.
3. Zero proj_head.params.grad — discard contamination before the proj backward.
4. Proj backward — fresh graph via proj_head(embed.detach()); backbone untouched.
5. Both optimizer steps (proj then backbone), both scheduler steps.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .clip_oracle import CLIPOracle, orthogonal_project
from .clip_preprocess import RenormalizeForCLIP
from .losses import BackboneLoss, ProjHeadLoss
from .models import DebiasedClassifier


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _fmt(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


class Trainer:
    """
    Dual-optimizer trainer for ERM baseline and debiasing runs.

    Parameters
    ----------
    model    : DebiasedClassifier
    cfg      : DebiasingConfig
    oracle   : CLIPOracle (frozen); pass None for ERM mode
    v_t_hat  : (D,) unit concept direction; pass None for ERM mode
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
        self._device_type = self.device.type

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

        # ── Loss functions ────────────────────────────────────────────────────
        self.backbone_criterion = BackboneLoss(cfg)
        self.proj_criterion = ProjHeadLoss()

        # ── Optimizers ────────────────────────────────────────────────────────
        # Backbone and proj head are updated by separate optimizers so that
        # each loss only touches the parameters it is responsible for.
        self.optimizer_backbone = AdamW(
            model.backbone.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.optimizer_proj = AdamW(
            model.proj_head.parameters(),
            lr=cfg.lr_proj,
            weight_decay=cfg.weight_decay,
        )

        # Separate AMP scalers — one per optimizer to avoid cross-scaler inf
        # tracking conflicts between the two backward passes.
        self.scaler_proj = GradScaler(self.device.type, enabled=cfg.amp)
        self.scaler_backbone = GradScaler(self.device.type, enabled=cfg.amp)

        # Schedulers set up once we know steps_per_epoch
        self.scheduler_backbone: OneCycleLR | None = None
        self.scheduler_proj: OneCycleLR | None = None

        # ── lambda_task warmup schedule ───────────────────────────────────────
        self._global_step: int = 0
        self._lambda_task_warmup_steps: int = 0       # set in _setup_schedulers
        self._lambda_task_0: float = cfg.lambda_task  # overwritten by _calibrate
        self._lambda_task_target: float = cfg.lambda_task

        self.ckpt_dir = os.path.join(
            cfg.checkpoint_dir, cfg.run_name, f"seed_{cfg.seed}"
        )
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self._best_val_acc = 0.0

    def _setup_schedulers(self, steps_per_epoch: int):
        if self.cfg.epochs <= 0:
            raise ValueError(
                f"Number of epochs must be positive, got {self.cfg.epochs}"
            )
        if steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be positive, got {steps_per_epoch}. Is the training dataset empty?"
            )

        total_steps = self.cfg.epochs * steps_per_epoch
        pct_start = self.cfg.warmup_epochs / self.cfg.epochs
        # OneCycleLR requires both phases to have ≥1 step; clamp away from 0 and 1.
        pct_start = max(1 / total_steps, min(1 - 1 / total_steps, pct_start))
        common = dict(
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )
        self.scheduler_backbone = OneCycleLR(
            self.optimizer_backbone, max_lr=self.cfg.lr, **common
        )
        self.scheduler_proj = OneCycleLR(
            self.optimizer_proj, max_lr=self.cfg.lr_proj, **common
        )

        if not self._erm_mode and self.cfg.lambda_task_warmup:
            self._lambda_task_warmup_steps = total_steps // 2

    # ── lambda_task warmup ────────────────────────────────────────────────────

    @torch.no_grad()
    def _calibrate_lambda_task(self, first_batch):
        """
        Single no-grad forward pass to set _lambda_task_0 such that
        lambda_task_0 * l_task_init ≈ min(active debiasing losses).
        """
        if self._erm_mode or not self.cfg.lambda_task_warmup:
            return

        images, labels, _ = first_batch
        images, labels = images.to(self.device), labels.to(self.device)

        clip_images = self.renorm(images)
        v_i = self.oracle.encode_images(clip_images)
        v_i_perp = F.normalize(orthogonal_project(v_i, self.v_t_hat), dim=-1)

        with autocast(self._device_type, enabled=self.cfg.amp):
            out = self.model(images)
            _, info = self.backbone_criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,
                v_t_hat=self.v_t_hat,
            )

        l_task = info["loss_task"]
        active = [
            info[k]
            for k, lam in [
                ("loss_align", self.cfg.lambda_align),
                ("loss_repulse", self.cfg.lambda_repulse),
            ]
            if lam > 0
        ]

        if not active or l_task < 1e-8:
            print("  [lambda_task warmup] calibration skipped (no active debiasing or l_task≈0)")
            return

        debias_ref = min(active)
        self._lambda_task_0 = debias_ref / l_task
        print(
            f"  [lambda_task warmup] schedule={self.cfg.lambda_task_warmup_schedule}"
            f"  l_task_init={l_task:.4f}  debias_ref={debias_ref:.4f}"
            f"  lambda_task_0={self._lambda_task_0:.4f}"
            f"  → {self._lambda_task_target:.4f} over {self._lambda_task_warmup_steps} steps"
        )

    def _current_lambda_task(self) -> float:
        if self._erm_mode or not self.cfg.lambda_task_warmup:
            return self._lambda_task_target
        schedule = self.cfg.lambda_task_warmup_schedule
        if schedule not in ("linear", "cosine"):
            raise ValueError(f"Unknown lambda_task_warmup_schedule: {schedule!r}")
        ws = self._lambda_task_warmup_steps
        if ws == 0 or self._global_step >= ws:
            return self._lambda_task_target
        t = self._global_step / ws  # 0 → 1 over warmup window
        if schedule == "cosine":
            t = (1.0 - math.cos(math.pi * t)) / 2.0  # half-cosine S-curve
        return self._lambda_task_0 + t * (self._lambda_task_target - self._lambda_task_0)

    # ── Single training step ──────────────────────────────────────────────────

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer_proj.zero_grad(set_to_none=True)
        self.optimizer_backbone.zero_grad(set_to_none=True)

        # CLIP targets — skipped in ERM mode
        v_i = None  # full CLIP image embedding  (proj head target)
        v_i_perp = None  # concept-scrubbed embedding (backbone target)

        if not self._erm_mode:
            clip_images = self.renorm(images)
            with torch.no_grad():
                v_i = self.oracle.encode_images(clip_images)  # (B, D)
                v_i_perp = F.normalize(
                    orthogonal_project(v_i, self.v_t_hat), dim=-1
                )  # (B, D)

        with autocast(self._device_type, enabled=self.cfg.amp):
            out = self.model(images)  # logits, embed, proj

        info = {}

        # ── Backward pass 1: backbone ─────────────────────────────────────────
        # Graph is consumed here.  L_align/L_repulse flow through out["proj"]
        # → proj_head, leaving contamination grads on proj_head.params.
        self.backbone_criterion.lambda_task = self._current_lambda_task()
        with autocast(self._device_type, enabled=self.cfg.amp):
            loss_backbone, backbone_info = self.backbone_criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,
                v_t_hat=self.v_t_hat,
            )

        self.scaler_backbone.scale(loss_backbone).backward()
        info.update(backbone_info)
        info["lambda_task"] = self.backbone_criterion.lambda_task

        # ── Backward pass 2: proj head ────────────────────────────────────────
        # Zero contamination grads before the proj backward so only
        # L_reconstruct grads remain on proj_head.params after this pass.
        # Uses a fresh graph (embed.detach()) — backbone params untouched.
        if not self._erm_mode:
            for p in self.model.proj_head.parameters():
                p.grad = None

            with autocast(self._device_type, enabled=self.cfg.amp):
                proj_for_loss = self.model.proj_head(out["embed"].detach())
                loss_proj, proj_info = self.proj_criterion(proj_for_loss, v_i)

            self.scaler_proj.scale(loss_proj).backward()
            info.update(proj_info)

        # ── Optimizer steps — both after all backward passes ─────────────────
        if not self._erm_mode:
            self.scaler_proj.unscale_(self.optimizer_proj)
            nn.utils.clip_grad_norm_(self.model.proj_head.parameters(), max_norm=1.0)
            self.scaler_proj.step(self.optimizer_proj)
            self.scaler_proj.update()

        self.scaler_backbone.unscale_(self.optimizer_backbone)
        nn.utils.clip_grad_norm_(self.model.backbone.parameters(), max_norm=1.0)
        self.scaler_backbone.step(self.optimizer_backbone)
        self.scaler_backbone.update()

        if self.scheduler_backbone is not None:
            self.scheduler_backbone.step()
        if self.scheduler_proj is not None and not self._erm_mode:
            self.scheduler_proj.step()

        self._global_step += 1
        return info, _accuracy(out["logits"], labels)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_correct = total = 0
        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with autocast(self._device_type, enabled=self.cfg.amp):
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
        best_ckpt_path : str
        """
        self._setup_schedulers(len(train_loader))
        self.model.to(self.device).train()
        first_batch = next(iter(train_loader))
        self._calibrate_lambda_task(first_batch)

        mode_str = (
            "ERM"
            if self._erm_mode
            else (
                f"debias  λ_align={self.cfg.lambda_align}  "
                f"λ_repulse={self.cfg.lambda_repulse}  "
                f"lr={self.cfg.lr}  lr_proj={self.cfg.lr_proj}"
            )
        )
        print(f"\n[{self.cfg.run_name}]  seed={self.cfg.seed}  {mode_str}")

        log_keys = (
            ["loss_task", "loss_backbone", "acc"]
            if self._erm_mode
            else [
                "loss_reconstruct",
                "loss_task",
                "loss_align",
                "loss_repulse",
                "loss_backbone",
                "lambda_task",
                "acc",
            ]
        )

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            running = {k: 0.0 for k in log_keys}
            n = 0

            for step, (images, labels, _) in enumerate(train_loader, 1):
                info, acc = self._train_step(images, labels)
                for k in log_keys:
                    if k == "acc":
                        running["acc"] += acc
                    elif k in info:
                        running[k] += info[k]
                n += 1

                if step % self.cfg.log_interval == 0:
                    avg = {k: v / n for k, v in running.items()}
                    lr = self.optimizer_backbone.param_groups[0]["lr"]
                    print(
                        f"  [ep {epoch}/{self.cfg.epochs}  "
                        f"step {step}/{len(train_loader)}]"
                        f"  lr={lr:.2e}  " + _fmt(avg)
                    )

            val = self._validate(val_loader)
            avg = {k: v / n for k, v in running.items()}
            print(f"\n  === Epoch {epoch} ({time.time()-t0:.1f}s) ===")
            print(f"    train — " + _fmt(avg))
            print(f"    val   — " + _fmt(val))

            ckpt = os.path.join(self.ckpt_dir, f"epoch_{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_backbone": self.optimizer_backbone.state_dict(),
                    "opt_proj": self.optimizer_proj.state_dict(),
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
