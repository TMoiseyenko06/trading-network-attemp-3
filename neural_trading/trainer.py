"""Walk-forward training engine with GPU optimization."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Optional
import time
import sys

from .model import NeuralOHLCVNet
from .loss import TradingLoss
from .gpu import GPUProfile, detect_gpu, scale_dim
from .preprocessing import compute_features, add_time_features, triple_barrier_labels, build_sequences


class WalkForwardTrainer:
    """Walk-forward training: train on history, validate on next period, roll forward."""

    def __init__(
        self,
        gpu_profile: Optional[GPUProfile] = None,
        lookback: int = 90,
        tp_pct: float = 0.0035,
        sl_pct: float = 0.002,
        max_bars: int = 20,
        lr: float = 1e-3,
        epochs_per_fold: int = 30,
        patience: int = 7,
    ):
        self.gpu = gpu_profile or detect_gpu()
        self.lookback = lookback
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.lr = lr
        self.epochs_per_fold = epochs_per_fold
        self.patience = patience

        # Scale model dimensions based on GPU
        self.hidden_dim = scale_dim(64, self.gpu.model_scale)
        self.device = self.gpu.device

        print(f"GPU: {self.gpu.name} ({self.gpu.total_memory_gb} GB)")
        print(f"  AMP: {self.gpu.use_amp} | TF32: {self.gpu.use_tf32} | "
              f"Compile: {self.gpu.use_compile}")
        print(f"  Batch size: {self.gpu.batch_size} | Hidden dim: {self.hidden_dim} | "
              f"Workers: {self.gpu.num_workers}")

    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess raw OHLCV dataframe into sequences."""
        features = compute_features(df)
        features = add_time_features(features)
        labels = triple_barrier_labels(df["close"], self.tp_pct, self.sl_pct, self.max_bars)

        # Drop initial NaN rows
        valid_start = features.first_valid_index()
        features = features.loc[valid_start:]
        labels = labels.loc[valid_start:]
        features = features.fillna(0)

        return build_sequences(features, labels, self.lookback)

    def _make_loader(
        self, X: np.ndarray, y_cls: np.ndarray, y_mag: np.ndarray, shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y_cls.astype(np.int64)),
            torch.from_numpy(y_mag),
        )
        return DataLoader(
            dataset,
            batch_size=self.gpu.batch_size,
            shuffle=shuffle,
            num_workers=self.gpu.num_workers,
            pin_memory=self.gpu.pin_memory,
            persistent_workers=self.gpu.num_workers > 0,
        )

    def _build_model(self, input_dim: int) -> NeuralOHLCVNet:
        model = NeuralOHLCVNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            conv_blocks=3,
            attn_heads=4,
            lstm_layers=2,
            num_classes=3,
        ).to(self.device)

        if self.gpu.use_compile:
            model = torch.compile(model)

        return model

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: TradingLoss,
        scaler: Optional[torch.amp.GradScaler],
    ) -> dict[str, float]:
        model.train()
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for step, (X, y_cls, y_mag) in enumerate(loader):
            X = X.to(self.device, non_blocking=True)
            y_cls = y_cls.to(self.device, non_blocking=True)
            y_mag = y_mag.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, pred_mag = model(X)
                loss, metrics = criterion(dir_logits, conf, pred_mag, y_cls, y_mag)
                loss = loss / self.gpu.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % self.gpu.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % self.gpu.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: TradingLoss,
    ) -> dict[str, float]:
        model.eval()
        total_metrics: dict[str, float] = {}
        n_batches = 0

        for X, y_cls, y_mag in loader:
            X = X.to(self.device, non_blocking=True)
            y_cls = y_cls.to(self.device, non_blocking=True)
            y_mag = y_mag.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, pred_mag = model(X)
                _, metrics = criterion(dir_logits, conf, pred_mag, y_cls, y_mag)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_months: int = 3,
        val_months: int = 1,
        test_months: int = 1,
    ) -> list[dict]:
        """Run walk-forward training across the full dataset.

        Args:
            df: Raw OHLCV dataframe with DatetimeIndex
            train_months: months of data for training each fold
            val_months: months for validation
            test_months: months for test (out-of-sample)

        Returns:
            List of per-fold test results
        """
        X_all, y_cls_all, y_mag_all = self._prepare_data(df)
        input_dim = X_all.shape[2]
        total_bars = len(X_all)

        # Estimate bars per month from the data
        if isinstance(df.index, pd.DatetimeIndex):
            total_days = (df.index[-1] - df.index[0]).days
            bars_per_day = total_bars / max(total_days, 1)
            bars_per_month = int(bars_per_day * 30)
        else:
            bars_per_month = total_bars // 6  # fallback

        train_size = bars_per_month * train_months
        val_size = bars_per_month * val_months
        test_size = bars_per_month * test_months
        fold_step = bars_per_month * (val_months + test_months)

        results = []
        fold = 0
        start = 0

        while start + train_size + val_size + test_size <= total_bars:
            t_end = start + train_size
            v_end = t_end + val_size
            te_end = v_end + test_size

            print(f"\n{'='*60}")
            print(f"Fold {fold}: train[{start}:{t_end}] val[{t_end}:{v_end}] test[{v_end}:{te_end}]")
            print(f"{'='*60}")

            train_loader = self._make_loader(
                X_all[start:t_end], y_cls_all[start:t_end], y_mag_all[start:t_end],
            )
            val_loader = self._make_loader(
                X_all[t_end:v_end], y_cls_all[t_end:v_end], y_mag_all[t_end:v_end],
                shuffle=False,
            )
            test_loader = self._make_loader(
                X_all[v_end:te_end], y_cls_all[v_end:te_end], y_mag_all[v_end:te_end],
                shuffle=False,
            )

            model = self._build_model(input_dim)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs_per_fold,
            )
            criterion = TradingLoss()
            scaler = torch.amp.GradScaler("cuda") if self.gpu.use_amp else None

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(self.epochs_per_fold):
                t0 = time.time()
                train_m = self._train_epoch(model, train_loader, optimizer, criterion, scaler)
                val_m = self._eval_epoch(model, val_loader, criterion)
                scheduler.step()
                elapsed = time.time() - t0

                val_loss = val_m["cls_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                print(
                    f"  Epoch {epoch:3d} | "
                    f"train_acc={train_m['accuracy']:.3f} val_acc={val_m['accuracy']:.3f} | "
                    f"sortino={val_m['sortino']:.3f} | {elapsed:.1f}s"
                )

                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # Load best model and evaluate on test set
            if best_state is not None:
                model.load_state_dict(best_state)
            test_m = self._eval_epoch(model, test_loader, criterion)
            print(f"  TEST | acc={test_m['accuracy']:.3f} sortino={test_m['sortino']:.3f}")

            results.append({
                "fold": fold,
                "test_accuracy": test_m["accuracy"],
                "test_sortino": test_m["sortino"],
                "best_val_loss": best_val_loss,
            })

            # Save fold model
            torch.save(
                {"model_state": best_state, "fold": fold, "hidden_dim": self.hidden_dim,
                 "input_dim": input_dim, "test_metrics": test_m},
                f"model_fold_{fold}.pt",
            )

            start += fold_step
            fold += 1

        return results
