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


def fixed_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    tp_points: float = 35.0,
    sl_points: float = 20.0,
    max_bars: int = 20,
) -> pd.DataFrame:
    """Compute labels using fixed point-based TP/SL barriers.

    For each bar, independently simulates a LONG and SHORT trade:
      - LONG: TP at entry + tp_points, SL at entry - sl_points
      - SHORT: TP at entry - tp_points, SL at entry + sl_points

    Labels based on which direction's TP gets hit first within max_bars.
    If neither TP is hit (only SLs or timeout) → flat.

    Returns:
        DataFrame with columns: label, magnitude
    """
    closes = close.values
    highs = high.values
    lows = low.values
    n = len(closes)

    labels = np.full(n, 2, dtype=np.int64)
    magnitudes = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry = closes[i]
        if entry == 0:
            continue

        long_tp = entry + tp_points
        long_sl = entry - sl_points
        short_tp = entry - tp_points
        short_sl = entry + sl_points

        long_outcome = 0   # 0=pending, 1=tp_hit, -1=sl_hit
        short_outcome = 0
        long_hit_bar = max_bars + 1
        short_hit_bar = max_bars + 1

        end = min(i + max_bars + 1, n)
        for j in range(1, end - i):
            h = highs[i + j]
            l = lows[i + j]

            # LONG trade resolution (check SL first — conservative)
            if long_outcome == 0:
                if l <= long_sl:
                    long_outcome = -1
                    long_hit_bar = j
                elif h >= long_tp:
                    long_outcome = 1
                    long_hit_bar = j

            # SHORT trade resolution (check SL first — conservative)
            if short_outcome == 0:
                if h >= short_sl:
                    short_outcome = -1
                    short_hit_bar = j
                elif l <= short_tp:
                    short_outcome = 1
                    short_hit_bar = j

            if long_outcome != 0 and short_outcome != 0:
                break

        # Assign label: whichever direction's TP hit first
        if long_outcome == 1 and (short_outcome != 1 or long_hit_bar <= short_hit_bar):
            labels[i] = 0  # LONG
            magnitudes[i] = tp_points / entry
        elif short_outcome == 1 and (long_outcome != 1 or short_hit_bar < long_hit_bar):
            labels[i] = 1  # SHORT
            magnitudes[i] = tp_points / entry
        else:
            labels[i] = 2  # FLAT — neither TP hit
            final_ret = abs(closes[min(i + max_bars, n - 1)] - entry) / entry
            magnitudes[i] = final_ret

    return pd.DataFrame(
        {"label": labels, "magnitude": magnitudes},
        index=close.index,
    )


def build_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flat arrays + lookback for lazy sequence building.

    Instead of materializing (N, lookback, features) which needs ~10GB+ RAM
    for 2.5M bars, we return the flat feature array and let the DataLoader
    build sequences on-the-fly on GPU.

    Returns:
        feat: (N, num_features) flat feature array
        y_class: (N,) int labels (aligned to feat, first `lookback-1` are padding)
        y_magnitude: (N,) float
        y_tp: (N,) float
        y_sl: (N,) float
    """
    feat = features.values.astype(np.float32)
    lab = labels["label"].values.astype(np.int64)
    mag = labels["magnitude"].values.astype(np.float32)

    has_tp_sl = "target_tp" in labels.columns and "target_sl" in labels.columns
    if has_tp_sl:
        tp = labels["target_tp"].values.astype(np.float32)
        sl = labels["target_sl"].values.astype(np.float32)
    else:
        tp = np.zeros(len(lab), dtype=np.float32)
        sl = np.zeros(len(lab), dtype=np.float32)

    # Replace any remaining NaN with 0 in features
    nan_count = np.isnan(feat).sum()
    if nan_count > 0:
        print(f"  Replacing {nan_count} NaN values in features with 0")
        np.nan_to_num(feat, copy=False)

    n = len(feat) - lookback
    if n <= 0:
        raise ValueError(f"Not enough data: {len(feat)} rows with lookback={lookback}")

    print(f"  Sequences: {n + 1} samples (lazy, {feat.shape[1]} features × {lookback} lookback)")

    return feat, lab, mag, tp, sl
