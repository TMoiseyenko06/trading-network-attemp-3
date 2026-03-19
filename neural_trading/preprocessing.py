"""Input preprocessing: stationary features from raw OHLCV + triple barrier labels."""

import numpy as np
import pandas as pd
from typing import Optional


def compute_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Convert raw OHLCV into stationary features.

    Input columns: open, high, low, close, volume
    Output columns: ret_open, ret_high, ret_low, ret_close,
                    range_pct, upper_wick, lower_wick, vol_zscore
    """
    out = pd.DataFrame(index=df.index)

    # Bar-to-bar percentage returns for OHLC
    for col in ["open", "high", "low", "close"]:
        out[f"ret_{col}"] = df[col].pct_change()

    # Bar range as percentage of close
    out["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Wick ratios (relative to bar range)
    bar_range = df["high"] - df["low"]
    bar_range_safe = bar_range.replace(0, np.nan)
    out["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / bar_range_safe
    out["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / bar_range_safe
    out["upper_wick"] = out["upper_wick"].fillna(0)
    out["lower_wick"] = out["lower_wick"].fillna(0)

    # Z-scored volume
    vol_mean = df["volume"].rolling(vol_window, min_periods=1).mean()
    vol_std = df["volume"].rolling(vol_window, min_periods=1).std().replace(0, 1)
    out["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-of-day and day-of-week features for regime conditioning."""
    out = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return out

    # Time of day as fraction [0, 1)
    minutes = df.index.hour * 60 + df.index.minute
    total = 24 * 60
    out["time_sin"] = np.sin(2 * np.pi * minutes / total)
    out["time_cos"] = np.cos(2 * np.pi * minutes / total)

    # Day of week as fraction [0, 1)
    dow = df.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    return out


def triple_barrier_labels(
    close: pd.Series,
    tp_pct: float = 0.0035,
    sl_pct: float = 0.002,
    max_bars: int = 20,
) -> pd.DataFrame:
    """Compute triple barrier labels aligned with trading SL/TP logic.

    For each bar, look forward up to max_bars and determine which barrier
    is hit first: take-profit (class 0), stop-loss (class 1), or timeout (class 2).

    Also returns the magnitude (max favorable excursion as pct).
    """
    closes = close.values
    n = len(closes)
    labels = np.full(n, 2, dtype=np.int64)      # default: timeout
    magnitudes = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry = closes[i]
        if entry == 0:
            continue
        best_move = 0.0
        for j in range(1, min(max_bars + 1, n - i)):
            ret = (closes[i + j] - entry) / entry
            best_move = max(best_move, abs(ret))
            if ret >= tp_pct:
                labels[i] = 0  # winner
                break
            elif ret <= -sl_pct:
                labels[i] = 1  # loser
                break
        magnitudes[i] = best_move

    result = pd.DataFrame(
        {"label": labels, "magnitude": magnitudes},
        index=close.index,
    )
    return result


def build_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences for training.

    Returns:
        X: (N, lookback, num_features)
        y_class: (N,) int labels
        y_magnitude: (N,) float magnitudes
    """
    feat = features.values.astype(np.float32)
    lab = labels["label"].values
    mag = labels["magnitude"].values.astype(np.float32)

    n = len(feat) - lookback
    if n <= 0:
        raise ValueError(f"Not enough data: {len(feat)} rows with lookback={lookback}")

    X = np.lib.stride_tricks.sliding_window_view(feat, lookback, axis=0)
    # sliding_window_view gives (N, features, lookback) — transpose to (N, lookback, features)
    X = np.moveaxis(X, -1, 1).copy()

    y_class = lab[lookback:].copy()
    y_mag = mag[lookback:].copy()

    # Drop any rows with NaN in features
    valid = ~np.isnan(X.reshape(X.shape[0], -1)).any(axis=1)
    return X[valid], y_class[valid], y_mag[valid]
