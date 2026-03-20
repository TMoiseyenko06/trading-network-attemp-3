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


def dynamic_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    max_bars: int = 20,
) -> pd.DataFrame:
    """Compute labels with optimal TP/SL targets derived from forward price action.

    Instead of fixed TP/SL percentages, this measures the actual max favorable
    excursion (MFE) and max adverse excursion (MAE) over the forward window,
    then labels based on the realized outcome.

    Returns:
        DataFrame with columns:
            label: 0=winner (MFE > MAE), 1=loser (MAE > MFE), 2=flat (tiny move)
            magnitude: absolute return at end of window
            target_tp: MFE as pct of entry — what TP should have been
            target_sl: MAE as pct of entry — what SL should have been
    """
    closes = close.values
    highs = high.values
    lows = low.values
    n = len(closes)

    labels = np.full(n, 2, dtype=np.int64)
    magnitudes = np.zeros(n, dtype=np.float64)
    target_tp = np.zeros(n, dtype=np.float64)
    target_sl = np.zeros(n, dtype=np.float64)

    flat_threshold = 0.0006  # ~15 NQ pts — filters noise, keeps meaningful moves

    for i in range(n):
        entry = closes[i]
        if entry == 0:
            continue

        end = min(i + max_bars + 1, n)
        if end <= i + 1:
            continue

        # Max favorable excursion (best high above entry)
        mfe = (highs[i + 1:end].max() - entry) / entry
        # Max adverse excursion (worst low below entry)
        mae = (entry - lows[i + 1:end].min()) / entry

        # Clamp to non-negative
        mfe = max(mfe, 0.0)
        mae = max(mae, 0.0)

        # Hard cap both TP and SL at 50 points (as pct of entry),
        # but also cap the pct itself to prevent huge values on low-priced instruments
        max_pct = min(50.0 / entry, 0.05)
        target_tp[i] = min(mfe, max_pct)
        target_sl[i] = min(mae, max_pct)

        # Realized return at end of window
        final_ret = (closes[min(i + max_bars, n - 1)] - entry) / entry
        magnitudes[i] = abs(final_ret)

        # Label: if the move is too small, it's flat
        if mfe < flat_threshold and mae < flat_threshold:
            labels[i] = 2  # flat
        elif mfe > mae:
            labels[i] = 0  # winner — favorable move dominated
        else:
            labels[i] = 1  # loser — adverse move dominated

    return pd.DataFrame(
        {
            "label": labels,
            "magnitude": magnitudes,
            "target_tp": target_tp,
            "target_sl": target_sl,
        },
        index=close.index,
    )


def build_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences for training.

    Returns:
        X: (N, lookback, num_features)
        y_class: (N,) int labels
        y_magnitude: (N,) float magnitudes
        y_tp: (N,) float target take-profit pct
        y_sl: (N,) float target stop-loss pct
    """
    feat = features.values.astype(np.float32)
    lab = labels["label"].values
    mag = labels["magnitude"].values.astype(np.float32)

    has_tp_sl = "target_tp" in labels.columns and "target_sl" in labels.columns
    if has_tp_sl:
        tp = labels["target_tp"].values.astype(np.float32)
        sl = labels["target_sl"].values.astype(np.float32)
    else:
        tp = np.zeros(len(lab), dtype=np.float32)
        sl = np.zeros(len(lab), dtype=np.float32)

    n = len(feat) - lookback
    if n <= 0:
        raise ValueError(f"Not enough data: {len(feat)} rows with lookback={lookback}")

    X = np.lib.stride_tricks.sliding_window_view(feat, lookback, axis=0)
    # sliding_window_view gives (N, features, lookback) — transpose to (N, lookback, features)
    X = np.moveaxis(X, -1, 1).copy()

    y_class = lab[lookback - 1:].copy()
    y_mag = mag[lookback - 1:].copy()
    y_tp = tp[lookback - 1:].copy()
    y_sl = sl[lookback - 1:].copy()

    # Drop any rows with NaN in features
    valid = ~np.isnan(X.reshape(X.shape[0], -1)).any(axis=1)
    return X[valid], y_class[valid], y_mag[valid], y_tp[valid], y_sl[valid]
