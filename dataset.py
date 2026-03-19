"""
PyTorch Dataset / DataLoader utilities for walk-forward training.

Handles windowed sequence creation, proper time-series splitting (no shuffle),
and GPU-optimal DataLoader configuration.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import GPUProfile


class OHLCVDataset(Dataset):
    """
    Dataset that serves (feature_window, direction_label, magnitude_label, close_return)
    tuples from pre-computed numpy arrays.

    Parameters
    ----------
    features       : (T, F) float32 — preprocessed feature matrix
    direction_labels : (T,) int64  — triple-barrier labels (0/1/2)
    magnitude_labels : (T,) float32 — signed move at barrier touch
    close_returns    : (T,) float32 — bar-to-bar close return (for P&L sim in loss)
    lookback         : int           — number of bars per input window
    """

    def __init__(
        self,
        features: np.ndarray,
        direction_labels: np.ndarray,
        magnitude_labels: np.ndarray,
        close_returns: np.ndarray,
        lookback: int = 90,
    ):
        assert len(features) == len(direction_labels) == len(magnitude_labels) == len(close_returns)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.directions = torch.tensor(direction_labels, dtype=torch.long)
        self.magnitudes = torch.tensor(magnitude_labels, dtype=torch.float32)
        self.returns = torch.tensor(close_returns, dtype=torch.float32)
        self.lookback = lookback
        self.length = len(features) - lookback

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        window = self.features[idx : idx + self.lookback]
        target_idx = idx + self.lookback
        return (
            window,
            self.directions[target_idx],
            self.magnitudes[target_idx],
            self.returns[target_idx],
        )


def make_loader(
    dataset: OHLCVDataset,
    profile: GPUProfile,
    shuffle: bool = False,
    batch_size_override: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader tuned to the GPU profile.

    Time-series data should NEVER be shuffled during training — the order
    matters for the drawdown / Sortino loss components.  Set shuffle=False.
    """
    bs = batch_size_override or profile.batch_size

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        prefetch_factor=profile.prefetch_factor if profile.num_workers > 0 else None,
        drop_last=False,
        persistent_workers=profile.num_workers > 0,
    )


def walk_forward_splits(
    total_bars: int,
    train_months: int = 4,
    val_months: int = 1,
    test_months: int = 1,
    bars_per_month: int = 8_400,    # ~21 trading days × 400 1-min bars
    step_months: int = 1,
) -> list[dict[str, tuple[int, int]]]:
    """
    Generate walk-forward (train, val, test) index ranges.

    Returns a list of dicts, each with keys 'train', 'val', 'test',
    values being (start_idx, end_idx) tuples.
    """
    window = (train_months + val_months + test_months) * bars_per_month
    step = step_months * bars_per_month
    splits = []

    start = 0
    while start + window <= total_bars:
        train_end = start + train_months * bars_per_month
        val_end = train_end + val_months * bars_per_month
        test_end = val_end + test_months * bars_per_month

        splits.append({
            "train": (start, train_end),
            "val": (train_end, val_end),
            "test": (val_end, test_end),
        })
        start += step

    return splits
