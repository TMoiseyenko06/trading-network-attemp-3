"""
Triple-barrier labeling for the target variable.

Given the close price at bar t, look forward and determine which event
happens first:
  0 = STOP LOSS  — price drops to (close - sl_points) before TP or time expiry
  1 = TAKE PROFIT — price rises to (close + tp_points) before SL or time expiry
  2 = TIMEOUT     — neither barrier hit within max_holding bars

This directly mirrors how you'd execute: SL at 20 pts, TP at 35 pts, etc.
The magnitude target is the actual price move (signed, in points) at the
moment the barrier is touched or time expires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit
def _triple_barrier_core(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tp_points: float,
    sl_points: float,
    max_holding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated inner loop.

    Returns
    -------
    direction_labels : int array  (0=SL, 1=TP, 2=Timeout)
    magnitude_labels : float array (signed move in points at barrier touch)
    """
    n = len(closes)
    directions = np.empty(n, dtype=np.int64)
    magnitudes = np.empty(n, dtype=np.float64)

    for i in range(n):
        entry = closes[i]
        upper = entry + tp_points
        lower = entry - sl_points
        label = 2          # default: timeout
        mag = 0.0

        horizon = min(i + max_holding, n)
        for j in range(i + 1, horizon):
            # Check SL first (conservative: if both hit same bar, count as loss)
            if lows[j] <= lower:
                label = 0
                mag = -sl_points
                break
            if highs[j] >= upper:
                label = 1
                mag = tp_points
                break

        if label == 2:
            # Timeout — magnitude is the move at expiry
            end_idx = min(i + max_holding, n - 1)
            mag = closes[end_idx] - entry

        directions[i] = label
        magnitudes[i] = mag

    return directions, magnitudes


def compute_labels(
    df: pd.DataFrame,
    tp_points: float = 35.0,
    sl_points: float = 20.0,
    max_holding: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute triple-barrier labels from a DataFrame with high, low, close.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'high', 'low', 'close' (case-insensitive).
    tp_points : float
        Take-profit distance in price points.
    sl_points : float
        Stop-loss distance in price points.
    max_holding : int
        Maximum bars to hold before timeout.

    Returns
    -------
    direction_labels : ndarray (int) — 0=SL, 1=TP, 2=Timeout
    magnitude_labels : ndarray (float) — signed move in points
    """
    cols = {c.lower().strip(): c for c in df.columns}
    highs = df[cols["high"]].values.astype(np.float64)
    lows = df[cols["low"]].values.astype(np.float64)
    closes = df[cols["close"]].values.astype(np.float64)

    directions, magnitudes = _triple_barrier_core(
        highs, lows, closes, tp_points, sl_points, max_holding,
    )
    return directions, magnitudes


def label_stats(directions: np.ndarray) -> dict[str, float]:
    """Quick summary of label distribution."""
    total = len(directions)
    return {
        "total": total,
        "sl_pct": (directions == 0).sum() / total * 100,
        "tp_pct": (directions == 1).sum() / total * 100,
        "timeout_pct": (directions == 2).sum() / total * 100,
    }
