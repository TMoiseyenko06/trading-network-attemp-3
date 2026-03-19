"""
Walk-forward training loop with:
  - Proper time-series splits (never shuffle)
  - Mixed-precision training (AMP) auto-scaled to GPU
  - Gradient accumulation for effective larger batches on smaller GPUs
  - Early stopping on validation loss
  - Per-fold checkpointing
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from config import GPUProfile
from model import TradingNetwork, build_model
from loss import TradingLoss
from dataset import OHLCVDataset, make_loader, walk_forward_splits
from preprocessing import NUM_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    lookback: int = 90
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 7              # early stopping patience (epochs)
    train_months: int = 4
    val_months: int = 1
    test_months: int = 1
    tp_points: float = 35.0
    sl_points: float = 20.0
    max_holding: int = 30
    checkpoint_dir: str = "checkpoints"


class Trainer:
    """Walk-forward trainer managing the full lifecycle."""

    def __init__(
        self,
        features: np.ndarray,          # (T, F) preprocessed
        direction_labels: np.ndarray,  # (T,) int
        magnitude_labels: np.ndarray,  # (T,) float
        close_returns: np.ndarray,     # (T,) float
        profile: GPUProfile,
        config: TrainConfig | None = None,
    ):
        self.features = features
        self.dir_labels = direction_labels
        self.mag_labels = magnitude_labels
        self.close_rets = close_returns
        self.profile = profile
        self.cfg = config or TrainConfig()
        self.ckpt_dir = Path(self.cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict]:
        """Execute full walk-forward training.  Returns per-fold metrics."""
        splits = walk_forward_splits(
            total_bars=len(self.features),
            train_months=self.cfg.train_months,
            val_months=self.cfg.val_months,
            test_months=self.cfg.test_months,
        )

        if not splits:
            logger.error("Not enough data for even one walk-forward split.")
            return []

        logger.info(f"Walk-forward: {len(splits)} folds")
        all_results = []

        for fold_idx, split in enumerate(splits):
            logger.info(f"\n{'='*60}\nFOLD {fold_idx+1}/{len(splits)}\n{'='*60}")
            result = self._train_fold(fold_idx, split)
            all_results.append(result)

        return all_results

    def _train_fold(self, fold_idx: int, split: dict) -> dict:
        """Train a single walk-forward fold."""
        tr_s, tr_e = split["train"]
        va_s, va_e = split["val"]
        te_s, te_e = split["test"]

        # Build datasets
        train_ds = OHLCVDataset(
            self.features[tr_s:tr_e], self.dir_labels[tr_s:tr_e],
            self.mag_labels[tr_s:tr_e], self.close_rets[tr_s:tr_e],
            lookback=self.cfg.lookback,
        )
        val_ds = OHLCVDataset(
            self.features[va_s:va_e], self.dir_labels[va_s:va_e],
            self.mag_labels[va_s:va_e], self.close_rets[va_s:va_e],
            lookback=self.cfg.lookback,
        )
        test_ds = OHLCVDataset(
            self.features[te_s:te_e], self.dir_labels[te_s:te_e],
            self.mag_labels[te_s:te_e], self.close_rets[te_s:te_e],
            lookback=self.cfg.lookback,
        )

        train_loader = make_loader(train_ds, self.profile, shuffle=False)
        val_loader = make_loader(val_ds, self.profile, shuffle=False)
        test_loader = make_loader(test_ds, self.profile, shuffle=False)

        # Fresh model each fold
        model = build_model(NUM_FEATURES, self.profile)
        criterion = TradingLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.max_epochs,
        )

        # AMP scaler (only for float16, bfloat16 doesn't need scaling)
        use_scaler = self.profile.use_amp and self.profile.amp_dtype == torch.float16
        scaler = GradScaler(enabled=use_scaler)

        best_val_loss = float("inf")
        patience_counter = 0
        best_ckpt = self.ckpt_dir / f"fold_{fold_idx}_best.pt"

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, scaler,
            )
            val_metrics = self._eval_epoch(model, val_loader, criterion)

            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_metrics['total']:.4f}  "
                f"val_loss={val_metrics['total']:.4f}  "
                f"val_ce={val_metrics['ce']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            # Early stopping
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        # Load best and evaluate on test
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))
        test_metrics = self._eval_epoch(model, test_loader, criterion)
        test_acc = self._accuracy(model, test_loader)

        logger.info(
            f"  TEST | loss={test_metrics['total']:.4f}  "
            f"ce={test_metrics['ce']:.4f}  acc={test_acc:.2%}"
        )

        return {
            "fold": fold_idx,
            "best_val_loss": best_val_loss,
            "test_loss": test_metrics["total"],
            "test_ce": test_metrics["ce"],
            "test_accuracy": test_acc,
            "checkpoint": str(best_ckpt),
        }

    def _train_epoch(
        self,
        model: TradingNetwork,
        loader,
        criterion: TradingLoss,
        optimizer,
        scaler: GradScaler,
    ) -> dict[str, float]:
        model.train()
        accum = self.profile.grad_accum_steps
        running = {}
        optimizer.zero_grad(set_to_none=True)

        for step, (x, dir_y, mag_y, ret_y) in enumerate(loader):
            x = x.to(self.profile.device, non_blocking=True)
            dir_y = dir_y.to(self.profile.device, non_blocking=True)
            mag_y = mag_y.to(self.profile.device, non_blocking=True)
            ret_y = ret_y.to(self.profile.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=self.profile.amp_dtype,
                enabled=self.profile.use_amp,
            ):
                d_logits, conf, mag = model(x)
                loss, metrics = criterion(d_logits, conf, mag, dir_y, mag_y, ret_y)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

        n = len(loader)
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: TradingNetwork,
        loader,
        criterion: TradingLoss,
    ) -> dict[str, float]:
        model.eval()
        running = {}

        for x, dir_y, mag_y, ret_y in loader:
            x = x.to(self.profile.device, non_blocking=True)
            dir_y = dir_y.to(self.profile.device, non_blocking=True)
            mag_y = mag_y.to(self.profile.device, non_blocking=True)
            ret_y = ret_y.to(self.profile.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=self.profile.amp_dtype,
                enabled=self.profile.use_amp,
            ):
                d_logits, conf, mag = model(x)
                _, metrics = criterion(d_logits, conf, mag, dir_y, mag_y, ret_y)

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v

        n = max(len(loader), 1)
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def _accuracy(self, model: TradingNetwork, loader) -> float:
        model.eval()
        correct = 0
        total = 0
        for x, dir_y, _, _ in loader:
            x = x.to(self.profile.device, non_blocking=True)
            dir_y = dir_y.to(self.profile.device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=self.profile.amp_dtype,
                enabled=self.profile.use_amp,
            ):
                d_logits, _, _ = model(x)

            preds = d_logits.argmax(dim=1)
            correct += (preds == dir_y).sum().item()
            total += len(dir_y)

        return correct / max(total, 1)
