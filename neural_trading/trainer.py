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
from .preprocessing import compute_features, add_time_features, dynamic_barrier_labels, build_sequences


class WalkForwardTrainer:
    """Walk-forward training: train on history, validate on next period, roll forward."""

    def __init__(
        self,
        gpu_profile: Optional[GPUProfile] = None,
        lookback: int = 90,
        max_bars: int = 20,
        lr: float = 1e-3,
        epochs_per_fold: int = 30,
        patience: int = 7,
    ):
        self.gpu = gpu_profile or detect_gpu()
        self.lookback = lookback
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

    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess raw OHLCV dataframe into sequences."""
        features = compute_features(df)
        features = add_time_features(features)
        labels = dynamic_barrier_labels(df["close"], df["high"], df["low"], self.max_bars)

        # Drop initial NaN rows
        valid_start = features.first_valid_index()
        features = features.loc[valid_start:]
        labels = labels.loc[valid_start:]

        # Replace inf/NaN before standardization — these come from
        # pct_change() on the first row and division-by-zero edge cases
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Standardize features (zero mean, unit variance) so the network
        # receives reasonably-scaled inputs instead of tiny pct-change values
        self._feat_mean = features.mean()
        self._feat_std = features.std().replace(0, 1)
        features = (features - self._feat_mean) / self._feat_std

        # Clip extreme outliers to prevent float16 overflow in AMP
        features = features.clip(-10, 10)

        # Diagnostics
        print(f"\n  Features ({features.shape[1]}): {list(features.columns)}")
        print(f"  Feature ranges after standardization:")
        for col in features.columns:
            vals = features[col].values
            print(f"    {col:15s}: mean={vals.mean():.4f} std={vals.std():.4f} "
                  f"min={vals.min():.4f} max={vals.max():.4f}")

        lbl_counts = np.bincount(labels["label"].values.astype(int), minlength=3)
        print(f"\n  Label distribution: 0(win)={lbl_counts[0]} 1(lose)={lbl_counts[1]} 2(flat)={lbl_counts[2]}")
        print(f"  Magnitude: mean={labels['magnitude'].mean():.6f} std={labels['magnitude'].std():.6f}")

        # Log-transform magnitude to tame heavy tails
        # (raw magnitudes have mean=137, std=2855 which blows up MSE in float16)
        labels = labels.copy()
        labels["magnitude"] = np.log1p(labels["magnitude"])
        # TP/SL targets: clip to model's output range so the loss doesn't chase
        # extreme MFE/MAE values the model can never output anyway
        from .model import DecisionHead
        if "target_tp" in labels.columns:
            labels["target_tp"] = labels["target_tp"].clip(
                DecisionHead.MIN_TP_PCT, DecisionHead.MAX_TP_PCT,
            )
            labels["target_sl"] = labels["target_sl"].clip(
                DecisionHead.MIN_SL_PCT, DecisionHead.MAX_SL_PCT,
            )

        return build_sequences(features, labels, self.lookback)

    def _make_loader(
        self, X: np.ndarray, y_cls: np.ndarray, y_mag: np.ndarray,
        y_tp: np.ndarray, y_sl: np.ndarray, shuffle: bool = True,
    ) -> DataLoader:
        # Move tensors to GPU upfront — avoids CPU→GPU transfers every batch
        # and eliminates the data-loading bottleneck entirely
        dev = self.device
        dataset = TensorDataset(
            torch.from_numpy(X).to(dev),
            torch.from_numpy(y_cls.astype(np.int64)).to(dev),
            torch.from_numpy(y_mag).to(dev),
            torch.from_numpy(y_tp).to(dev),
            torch.from_numpy(y_sl).to(dev),
        )
        return DataLoader(
            dataset,
            batch_size=self.gpu.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,  # already on GPU
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

        for step, (X, y_cls, y_mag, y_tp, y_sl) in enumerate(loader):
            # Data already on GPU from _make_loader
            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, pred_mag, pred_tp, pred_sl = model(X)
                loss, metrics = criterion(dir_logits, conf, pred_mag, pred_tp, pred_sl, y_cls, y_mag, y_tp, y_sl)
                loss = loss / self.gpu.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % self.gpu.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % self.gpu.gradient_accumulation_steps == 0:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # Log gradient norm on first batch of each epoch
            if step == 0:
                total_metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

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

        for X, y_cls, y_mag, y_tp, y_sl in loader:
            # Data already on GPU from _make_loader
            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, pred_mag, pred_tp, pred_sl = model(X)
                _, metrics = criterion(dir_logits, conf, pred_mag, pred_tp, pred_sl, y_cls, y_mag, y_tp, y_sl)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1

        return {k: v / n_batches for k, v in total_metrics.items()}

    def train_backtest(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.8,
        val_pct: float = 0.1,
    ) -> dict:
        """Train on first portion of data, backtest on the rest.

        Args:
            df: Raw OHLCV dataframe with DatetimeIndex
            train_pct: fraction of data for training (default 80%)
            val_pct: fraction of training data used for validation / early stopping

        Returns:
            Dict with backtest results
        """
        X_all, y_cls_all, y_mag_all, y_tp_all, y_sl_all = self._prepare_data(df)
        input_dim = X_all.shape[2]
        total = len(X_all)

        split = int(total * train_pct)
        val_size = int(split * val_pct)
        train_end = split - val_size

        # Compute class weights from training labels to handle imbalance
        train_labels = y_cls_all[:train_end]
        counts = np.bincount(train_labels.astype(int), minlength=3).astype(np.float32)
        print(f"\n  Label distribution (train): {dict(enumerate(counts.astype(int)))}")
        # Inverse frequency weighting
        counts = np.maximum(counts, 1.0)  # avoid div by zero
        class_weights = (1.0 / counts) * counts.sum() / len(counts)
        class_weights_t = torch.from_numpy(class_weights).to(self.device)
        print(f"  Class weights: {class_weights}")

        print(f"\n{'='*60}")
        print(f"Train[0:{train_end}] Val[{train_end}:{split}] Backtest[{split}:{total}]")
        print(f"  {train_end} train / {val_size} val / {total - split} backtest bars")
        print(f"{'='*60}")

        train_loader = self._make_loader(
            X_all[:train_end], y_cls_all[:train_end], y_mag_all[:train_end],
            y_tp_all[:train_end], y_sl_all[:train_end],
        )
        val_loader = self._make_loader(
            X_all[train_end:split], y_cls_all[train_end:split], y_mag_all[train_end:split],
            y_tp_all[train_end:split], y_sl_all[train_end:split],
            shuffle=False,
        )
        test_loader = self._make_loader(
            X_all[split:], y_cls_all[split:], y_mag_all[split:],
            y_tp_all[split:], y_sl_all[split:],
            shuffle=False,
        )

        model = self._build_model(input_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs_per_fold,
        )
        criterion = TradingLoss(class_weights=class_weights_t)
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

            gnorm = train_m.get("grad_norm", 0)
            rr = val_m.get("rr_ratio", 0)
            pnl = val_m.get("pnl", 0)
            flat = val_m.get("flat_rate", 0)
            trade_acc = val_m.get("trade_acc", 0)
            print(
                f"  Epoch {epoch:3d} | "
                f"val_acc={val_m['accuracy']:.3f} trade_acc={trade_acc:.3f} | "
                f"loss={val_m['cls_loss']:.4f} pnl={pnl:.5f} sortino={val_m['sortino']:.3f} "
                f"R:R={rr:.2f} flat={flat:.0%} | "
                f"gnorm={gnorm:.4f} | {elapsed:.1f}s"
            )

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best model and backtest
        if best_state is not None:
            model.load_state_dict(best_state)
        test_m = self._eval_epoch(model, test_loader, criterion)
        test_pnl = test_m.get("pnl", 0)
        test_flat = test_m.get("flat_rate", 0)
        test_trade_acc = test_m.get("trade_acc", 0)
        print(f"\n  BACKTEST | acc={test_m['accuracy']:.3f} trade_acc={test_trade_acc:.3f} "
              f"pnl={test_pnl:.5f} sortino={test_m['sortino']:.3f} flat={test_flat:.0%}")

        torch.save(
            {"model_state": best_state, "hidden_dim": self.hidden_dim,
             "input_dim": input_dim, "test_metrics": test_m,
             "feat_mean": self._feat_mean.to_dict(),
             "feat_std": self._feat_std.to_dict(),
             # Resume support
             "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict(),
             "scaler_state": scaler.state_dict() if scaler is not None else None,
             "epoch": epoch,
             "best_val_loss": best_val_loss,
             "lr": self.lr,
             "epochs_per_fold": self.epochs_per_fold,
             },
            "model.pt",
        )
        print("  Saved model.pt")

        return {
            "test_accuracy": test_m["accuracy"],
            "test_sortino": test_m["sortino"],
            "best_val_loss": best_val_loss,
        }

    def resume_training(
        self,
        df: pd.DataFrame,
        checkpoint_path: str = "model.pt",
        extra_epochs: int = 30,
        train_pct: float = 0.8,
        val_pct: float = 0.1,
    ) -> dict:
        """Resume training from a saved checkpoint.

        Args:
            df: Raw OHLCV dataframe with DatetimeIndex
            checkpoint_path: Path to the saved model.pt checkpoint
            extra_epochs: Number of additional epochs to train
            train_pct: fraction of data for training (default 80%)
            val_pct: fraction of training data used for validation / early stopping
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        prev_epoch = checkpoint.get("epoch", 0)
        prev_best_val = checkpoint.get("best_val_loss", float("inf"))

        print(f"  Resuming from epoch {prev_epoch + 1}, best_val_loss={prev_best_val:.4f}")
        print(f"  Training {extra_epochs} additional epochs")

        X_all, y_cls_all, y_mag_all, y_tp_all, y_sl_all = self._prepare_data(df)
        input_dim = X_all.shape[2]
        total = len(X_all)

        split = int(total * train_pct)
        val_size = int(split * val_pct)
        train_end = split - val_size

        # Class weights
        train_labels = y_cls_all[:train_end]
        counts = np.bincount(train_labels.astype(int), minlength=3).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        class_weights = (1.0 / counts) * counts.sum() / len(counts)
        class_weights_t = torch.from_numpy(class_weights).to(self.device)

        print(f"\n{'='*60}")
        print(f"RESUME TRAINING")
        print(f"Train[0:{train_end}] Val[{train_end}:{split}] Backtest[{split}:{total}]")
        print(f"  {train_end} train / {val_size} val / {total - split} backtest bars")
        print(f"{'='*60}")

        train_loader = self._make_loader(
            X_all[:train_end], y_cls_all[:train_end], y_mag_all[:train_end],
            y_tp_all[:train_end], y_sl_all[:train_end],
        )
        val_loader = self._make_loader(
            X_all[train_end:split], y_cls_all[train_end:split], y_mag_all[train_end:split],
            y_tp_all[train_end:split], y_sl_all[train_end:split],
            shuffle=False,
        )
        test_loader = self._make_loader(
            X_all[split:], y_cls_all[split:], y_mag_all[split:],
            y_tp_all[split:], y_sl_all[split:],
            shuffle=False,
        )

        # Rebuild model and load weights
        model = self._build_model(input_dim)
        state = checkpoint["model_state"]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)

        # Rebuild optimizer and load state
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        # Fresh cosine schedule for the extra epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=extra_epochs,
        )

        criterion = TradingLoss(class_weights=class_weights_t)
        scaler = torch.amp.GradScaler("cuda") if self.gpu.use_amp else None
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        best_val_loss = prev_best_val
        patience_counter = 0
        best_state = {k.removeprefix("_orig_mod."): v.cpu().clone()
                      for k, v in model.state_dict().items()}

        start_epoch = prev_epoch + 1
        for epoch in range(start_epoch, start_epoch + extra_epochs):
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

            gnorm = train_m.get("grad_norm", 0)
            rr = val_m.get("rr_ratio", 0)
            pnl = val_m.get("pnl", 0)
            flat = val_m.get("flat_rate", 0)
            trade_acc = val_m.get("trade_acc", 0)
            print(
                f"  Epoch {epoch:3d} | "
                f"val_acc={val_m['accuracy']:.3f} trade_acc={trade_acc:.3f} | "
                f"loss={val_m['cls_loss']:.4f} pnl={pnl:.5f} sortino={val_m['sortino']:.3f} "
                f"R:R={rr:.2f} flat={flat:.0%} | "
                f"gnorm={gnorm:.4f} | {elapsed:.1f}s"
            )

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best model and backtest
        if best_state is not None:
            model.load_state_dict(best_state)
        test_m = self._eval_epoch(model, test_loader, criterion)
        test_pnl = test_m.get("pnl", 0)
        test_flat = test_m.get("flat_rate", 0)
        test_trade_acc = test_m.get("trade_acc", 0)
        print(f"\n  BACKTEST | acc={test_m['accuracy']:.3f} trade_acc={test_trade_acc:.3f} "
              f"pnl={test_pnl:.5f} sortino={test_m['sortino']:.3f} flat={test_flat:.0%}")

        torch.save(
            {"model_state": best_state, "hidden_dim": self.hidden_dim,
             "input_dim": input_dim, "test_metrics": test_m,
             "feat_mean": self._feat_mean.to_dict(),
             "feat_std": self._feat_std.to_dict(),
             "optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict(),
             "scaler_state": scaler.state_dict() if scaler is not None else None,
             "epoch": epoch,
             "best_val_loss": best_val_loss,
             "lr": self.lr,
             "epochs_per_fold": extra_epochs,
             },
            checkpoint_path,
        )
        print(f"  Saved {checkpoint_path}")

        return {
            "test_accuracy": test_m["accuracy"],
            "test_sortino": test_m["sortino"],
            "best_val_loss": best_val_loss,
        }
