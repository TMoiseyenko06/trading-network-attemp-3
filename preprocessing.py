"""
Input preprocessing: convert raw OHLCV into stationary features.

All features are derived purely from OHLCV — no external indicators.
  - Percentage returns bar-to-bar for O, H, L, C
  - Z-scored volume (rolling mean/std)
  - Bar range as percentage of close
  - Upper / lower wick ratios
  - Time-of-day and day-of-week (sin/cos encoded for continuity)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Column ordering the model expects
FEATURE_COLS: list[str] = [
    "open_ret", "high_ret", "low_ret", "close_ret",
    "vol_zscore",
    "bar_range_pct",
    "upper_wick_ratio", "lower_wick_ratio",
    "tod_sin", "tod_cos",
    "dow_sin", "dow_cos",
]

NUM_FEATURES: int = len(FEATURE_COLS)


def preprocess(
    df: pd.DataFrame,
    vol_lookback: int = 20,
    clip_std: float = 5.0,
) -> pd.DataFrame:
    """
    Transform a DataFrame with columns [timestamp, open, high, low, close, volume]
    into the stationary feature set the network consumes.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: timestamp (or datetime index), open, high, low, close, volume.
    vol_lookback : int
        Rolling window for volume z-score (default 20 bars).
    clip_std : float
        Clip extreme values at ±clip_std standard deviations.

    Returns
    -------
    pd.DataFrame
        Cleaned feature frame indexed by timestamp, NaN rows dropped.
    """
    df = df.copy()

    # Ensure lowercase columns
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # ---- percentage returns (bar-to-bar) ---------------------------------
    for col in ["open", "high", "low", "close"]:
        df[f"{col}_ret"] = df[col].pct_change()

    # ---- volume z-score ---------------------------------------------------
    vol_mean = df["volume"].rolling(vol_lookback, min_periods=1).mean()
    vol_std = df["volume"].rolling(vol_lookback, min_periods=1).std().replace(0, 1)
    df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

    # ---- bar range as % of close ------------------------------------------
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # ---- wick ratios -------------------------------------------------------
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["upper_wick_ratio"] = (df["high"] - body_top) / bar_range
    df["lower_wick_ratio"] = (body_bot - df["low"]) / bar_range

    # ---- time-of-day / day-of-week (sin/cos) ------------------------------
    seconds_in_day = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
    tod_frac = seconds_in_day / 86400.0
    df["tod_sin"] = np.sin(2 * np.pi * tod_frac)
    df["tod_cos"] = np.cos(2 * np.pi * tod_frac)

    dow_frac = df.index.dayofweek / 7.0
    df["dow_sin"] = np.sin(2 * np.pi * dow_frac)
    df["dow_cos"] = np.cos(2 * np.pi * dow_frac)

    # ---- select & clean ----------------------------------------------------
    df = df[FEATURE_COLS].copy()
    df = df.dropna()

    # Clip outliers
    for col in FEATURE_COLS:
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma > 0:
            df[col] = df[col].clip(mu - clip_std * sigma, mu + clip_std * sigma)

    return df


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the feature matrix to produce (X, y) pairs.

    Parameters
    ----------
    features : ndarray of shape (T, F)
    labels   : ndarray of shape (T,) or (T, L) — aligned with features
    lookback : int — number of bars per sample

    Returns
    -------
    X : ndarray of shape (N, lookback, F)
    y : ndarray of shape (N,) or (N, L)
    """
    T = len(features)
    assert len(labels) == T, "features and labels must have the same length"

    N = T - lookback
    F = features.shape[1]

    X = np.empty((N, lookback, F), dtype=np.float32)
    for i in range(N):
        X[i] = features[i : i + lookback]

    y = labels[lookback:]
    return X, y
