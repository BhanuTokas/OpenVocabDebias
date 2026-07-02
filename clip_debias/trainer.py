"""
trainer.py — Dual-optimizer training loop with concept subspace support.

Two optimizers with separate responsibilities:

  optimizer_proj     — updates projection head only
                       loss: L_reconstruct = 1 - cos(P(E(x)), V_I)

  optimizer_backbone — updates backbone only
                       loss: λ_task·L_task + λ_align·L_align + λ_repulse·L_repulse
                       L_repulse now penalises ALL k concept subspace directions.

Subspace change summary
-----------------------
v_t_hat (D,) is replaced by subspace (k, D) throughout.
compute_distillation_targets() now returns both v_i AND v_i_perp.
BackboneLoss.forward() accepts subspace= instead of v_t_hat=.
_calibrate_lambdas uses the subspace-projected targets for calibration.

Step order per batch
--------------------
1. zero_grad both optimizers
2. Forward pass
3. Backbone backward (graph consumed)
4. Zero proj_head contamination grads
5. Proj backward (fresh graph via embed.detach())
6. Both optimizer steps
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

from .clip_oracle import CLIPOracle, compute_distillation_targets
from .clip_preprocess import RenormalizeForCLIP
from .evaluate import extract_features, train_linear_probe, worst_group_accuracy
from .losses import BackboneLoss, ProjHeadLoss
from .models import DebiasedClassifier


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _fmt(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


class Trainer:
    """
    Dual-optimizer trainer.

    Parameters
    ----------
    model    : DebiasedClassifier
    cfg      : DebiasingConfig
    oracle   : CLIPOracle (frozen); None for ERM mode
    subspace : (k, D) orthonormal concept subspace; None for ERM mode.
               Pass a (1, D) tensor to reproduce single-direction behaviour.
    """

    def __init__(
        self,
        model: DebiasedClassifier,
        cfg,
        oracle: CLIPOracle | None = None,
        subspace: torch.Tensor | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._device_type = self.device.type

        self._erm_mode = cfg.lambda_align == 0.0 and cfg.lambda_repulse == 0.0
        if not self._erm_mode and (oracle is None or subspace is None):
            raise ValueError(
                "oracle and subspace are required when lambda_align or "
                "lambda_repulse > 0.  Pass oracle=None only for ERM mode."
            )

        self.oracle = oracle
        self.subspace = subspace.to(self.device) if subspace is not None else None
        self.renorm = (
            RenormalizeForCLIP().to(self.device) if not self._erm_mode else None
        )

        self.backbone_criterion = BackboneLoss(cfg)
        self.proj_criterion = ProjHeadLoss()

        self.optimizer_backbone = AdamW(
            model.backbone.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.optimizer_proj = AdamW(
            model.proj_head.parameters(), lr=cfg.lr_proj, weight_decay=cfg.weight_decay
        )

        self.scaler_proj = GradScaler(self.device.type, enabled=cfg.amp)
        self.scaler_backbone = GradScaler(self.device.type, enabled=cfg.amp)

        self.scheduler_backbone: OneCycleLR | None = None
        self.scheduler_proj: OneCycleLR | None = None

        # ── Lambda schedules ──────────────────────────────────────────────────
        self._global_step: int = 0
        self._lambda_task_warmup_steps: int = 0
        self._lambda_align_warmup_steps: int = 0
        self._lambda_repulse_decay_steps: int = 0
        self._lambda_task_0: float = cfg.lambda_task
        self._lambda_task_target: float = cfg.lambda_task
        self._lambda_align_0: float = cfg.lambda_align
        self._lambda_align_target: float = cfg.lambda_align
        self._lambda_repulse_0: float = cfg.lambda_repulse
        self._lambda_repulse_target: float = cfg.lambda_repulse

        self.ckpt_dir = os.path.join(
            cfg.checkpoint_dir, cfg.run_name, f"seed_{cfg.seed}"
        )
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        self._best_val_acc = 0.0
        self._best_val_wga: float = -1.0

    # ── Scheduler setup ───────────────────────────────────────────────────────

    def _setup_schedulers(self, steps_per_epoch: int):
        if self.cfg.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.cfg.epochs}")
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")

        total_steps = self.cfg.epochs * steps_per_epoch
        pct_start = self.cfg.warmup_epochs / self.cfg.epochs
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
        if not self._erm_mode and self.cfg.lambda_align_warmup:
            self._lambda_align_warmup_steps = total_steps // 2
        if (
            not self._erm_mode
            and self.cfg.lambda_repulse_decay
            and self.cfg.lambda_repulse > 0
        ):
            self._lambda_repulse_decay_steps = total_steps // 2

    # ── Lambda calibration ────────────────────────────────────────────────────

    @torch.no_grad()
    def _calibrate_lambdas(self, first_batch):
        """
        Single no-grad forward pass to calibrate initial lambda values.
        Uses the subspace-projected V_I_perp for computing loss magnitudes.
        """
        need_task = not self._erm_mode and self.cfg.lambda_task_warmup
        need_align = (
            not self._erm_mode
            and self.cfg.lambda_align_warmup
            and self.cfg.lambda_align > 0
        )
        need_repulse = (
            not self._erm_mode
            and self.cfg.lambda_repulse_decay
            and self.cfg.lambda_repulse > 0
        )
        if not need_task and not need_align and not need_repulse:
            return

        images, labels, _ = first_batch
        images, labels = images.to(self.device), labels.to(self.device)

        clip_images = self.renorm(images)
        v_i, v_i_perp = compute_distillation_targets(
            self.oracle, clip_images, self.subspace, renormalize=True
        )

        with autocast(self._device_type, enabled=self.cfg.amp):
            out = self.model(images)
            _, info = self.backbone_criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,
                subspace=self.subspace,
            )

        l_task, l_align, l_repulse = (
            info["loss_task"],
            info["loss_align"],
            info["loss_repulse"],
        )

        if self.cfg.lambda_repulse > 0:
            anchor, anchor_name = l_repulse, "loss_repulse"
        elif self.cfg.lambda_align > 0:
            anchor, anchor_name = l_align, "loss_align"
        else:
            return

        if need_task and l_task > 1e-8:
            self._lambda_task_0 = anchor / l_task
            print(
                f"  [lambda_task warmup] schedule={self.cfg.lambda_task_warmup_schedule}"
                f"  anchor={anchor_name}={anchor:.4f}  l_task={l_task:.4f}"
                f"  lambda_task_0={self._lambda_task_0:.4f} → {self._lambda_task_target:.4f}"
                f"  over {self._lambda_task_warmup_steps} steps"
            )

        if need_align and self.cfg.lambda_repulse > 0 and l_align > 1e-8:
            self._lambda_align_0 = l_repulse / l_align
            print(
                f"  [lambda_align warmup] schedule={self.cfg.lambda_align_warmup_schedule}"
                f"  anchor=loss_repulse={l_repulse:.4f}  l_align={l_align:.4f}"
                f"  lambda_align_0={self._lambda_align_0:.4f} → {self._lambda_align_target:.4f}"
                f"  over {self._lambda_align_warmup_steps} steps"
            )

        if need_repulse and l_task > 1e-8 and l_repulse > 1e-8:
            self._lambda_repulse_0 = l_task / l_repulse
            print(
                f"  [lambda_repulse decay] schedule={self.cfg.lambda_repulse_decay_schedule}"
                f"  l_task={l_task:.4f}  l_repulse={l_repulse:.4f}"
                f"  lambda_repulse_0={self._lambda_repulse_0:.4f} → {self._lambda_repulse_target:.4f}"
                f"  over {self._lambda_repulse_decay_steps} steps"
            )

    # ── Lambda schedule helpers ───────────────────────────────────────────────

    def _cosine_interp(self, t: float) -> float:
        return (1.0 - math.cos(math.pi * t)) / 2.0

    def _scheduled_lambda(self, v0, vtarget, window_steps, schedule) -> float:
        if window_steps == 0 or self._global_step >= window_steps:
            return vtarget
        t = self._global_step / window_steps
        if schedule == "cosine":
            t = self._cosine_interp(t)
        elif schedule != "linear":
            raise ValueError(f"Unknown schedule: {schedule!r}")
        return v0 + t * (vtarget - v0)

    def _current_lambda_task(self) -> float:
        if self._erm_mode or not self.cfg.lambda_task_warmup:
            return self._lambda_task_target
        return self._scheduled_lambda(
            self._lambda_task_0,
            self._lambda_task_target,
            self._lambda_task_warmup_steps,
            self.cfg.lambda_task_warmup_schedule,
        )

    def _current_lambda_align(self) -> float:
        if self._erm_mode or not self.cfg.lambda_align_warmup:
            return self._lambda_align_target
        return self._scheduled_lambda(
            self._lambda_align_0,
            self._lambda_align_target,
            self._lambda_align_warmup_steps,
            self.cfg.lambda_align_warmup_schedule,
        )

    def _current_lambda_repulse(self) -> float:
        if self._erm_mode or not self.cfg.lambda_repulse_decay:
            return self._lambda_repulse_target
        return self._scheduled_lambda(
            self._lambda_repulse_0,
            self._lambda_repulse_target,
            self._lambda_repulse_decay_steps,
            self.cfg.lambda_repulse_decay_schedule,
        )

    # ── Single training step ──────────────────────────────────────────────────

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer_proj.zero_grad(set_to_none=True)
        self.optimizer_backbone.zero_grad(set_to_none=True)

        v_i = v_i_perp = None
        if not self._erm_mode:
            clip_images = self.renorm(images)
            with torch.no_grad():
                v_i, v_i_perp = compute_distillation_targets(
                    self.oracle, clip_images, self.subspace, renormalize=True
                )

        with autocast(self._device_type, enabled=self.cfg.amp):
            out = self.model(images)

        info = {}

        # ── Backbone backward ─────────────────────────────────────────────────
        self.backbone_criterion.lambda_task = self._current_lambda_task()
        self.backbone_criterion.lambda_align = self._current_lambda_align()
        self.backbone_criterion.lambda_repulse = self._current_lambda_repulse()
        with autocast(self._device_type, enabled=self.cfg.amp):
            loss_backbone, backbone_info = self.backbone_criterion(
                logits=out["logits"],
                labels=labels,
                proj=out["proj"],
                v_i_perp=v_i_perp,
                subspace=self.subspace,
            )
        self.scaler_backbone.scale(loss_backbone).backward()
        info.update(backbone_info)
        info["lambda_task"] = self.backbone_criterion.lambda_task
        info["lambda_align"] = self.backbone_criterion.lambda_align
        info["lambda_repulse_sched"] = self.backbone_criterion.lambda_repulse

        # ── Proj head backward ────────────────────────────────────────────────
        # Zero contamination grads left by backbone backward, then recompute
        # proj from a fresh graph (embed.detach()) so backbone is untouched.
        if not self._erm_mode:
            for p in self.model.proj_head.parameters():
                p.grad = None
            with autocast(self._device_type, enabled=self.cfg.amp):
                proj_for_loss = self.model.proj_head(out["embed"].detach())
                loss_proj, proj_info = self.proj_criterion(proj_for_loss, v_i)
            self.scaler_proj.scale(loss_proj).backward()
            info.update(proj_info)

        # ── Optimizer steps ───────────────────────────────────────────────────
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
        self._setup_schedulers(len(train_loader))
        self.model.to(self.device).train()
        first_batch = next(iter(train_loader))
        self._calibrate_lambdas(first_batch)

        k = self.subspace.shape[0] if self.subspace is not None else 0
        mode_str = (
            "ERM"
            if self._erm_mode
            else (
                f"debias  subspace_k={k}  λ_align={self.cfg.lambda_align}"
                f"  λ_repulse={self.cfg.lambda_repulse}"
                f"  lr={self.cfg.lr}  lr_proj={self.cfg.lr_proj}"
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
                "lambda_align",
                "lambda_repulse_sched",
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
                        f"  [ep {epoch}/{self.cfg.epochs}  step {step}/{len(train_loader)}]"
                        f"  lr={lr:.2e}  " + _fmt(avg)
                    )

            val = self._validate(val_loader)
            avg = {k: v / n for k, v in running.items()}
            print(f"\n  === Epoch {epoch} ({time.time()-t0:.1f}s) ===")
            print(f"    train — " + _fmt(avg))
            print(f"    val   — " + _fmt(val))

            train_feats = extract_features(
                self.model, train_loader, str(self.device), use_amp=self.cfg.amp
            )
            val_feats = extract_features(
                self.model, val_loader, str(self.device), use_amp=self.cfg.amp
            )
            probe = train_linear_probe(
                X_train=train_feats["embeds"],
                y_train=train_feats["concepts"],
                X_test=val_feats["embeds"],
                y_test=val_feats["concepts"],
            )
            print(
                f"    probe — "
                + _fmt(
                    {
                        "probe_train_acc": probe["probe_train_acc"],
                        "probe_val_acc": probe["probe_test_acc"],
                    }
                )
            )

            _, val_wga = worst_group_accuracy(
                targets=val_feats["targets"],
                concepts=val_feats["concepts"],
                preds=val_feats["preds"],
            )
            print(f"    val   — val_wga: {val_wga:.4f}")
            self.model.train()

            ckpt = os.path.join(self.ckpt_dir, f"epoch_{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "opt_backbone": self.optimizer_backbone.state_dict(),
                    "opt_proj": self.optimizer_proj.state_dict(),
                    "val_acc": val["val_acc"],
                    "val_wga": val_wga,
                    "run_name": self.cfg.run_name,
                    "seed": self.cfg.seed,
                    "subspace_k": (
                        self.subspace.shape[0] if self.subspace is not None else 0
                    ),
                },
                ckpt,
            )

            if val["val_acc"] > self._best_val_acc:
                self._best_val_acc = val["val_acc"]
                best = os.path.join(self.ckpt_dir, "best.pt")
                torch.save(self.model.state_dict(), best)
                print(f"    ✓ best val_acc={val['val_acc']:.4f}  → {best}")

            if val_wga > self._best_val_wga:
                self._best_val_wga = val_wga
                best_wga = os.path.join(self.ckpt_dir, "best_wga.pt")
                torch.save(self.model.state_dict(), best_wga)
                print(f"    ✓ best val_wga={val_wga:.4f}  → {best_wga}")

        print()
        return os.path.join(self.ckpt_dir, "best_wga.pt")
