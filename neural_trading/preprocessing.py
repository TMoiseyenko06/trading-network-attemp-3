"""Input preprocessing: stationary features from raw OHLCV + triple barrier labels."""

import numpy as np
import pandas as pd
from typing import Optional


def compute_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Convert raw OHLCV into stationary features.

    Input columns: open, high, low, close, volume
    Output columns:
      Bar structure (8):  ret_open, ret_high, ret_low, ret_close,
                          range_pct, upper_wick, lower_wick, vol_zscore
      Multi-TF returns (3): ret_5, ret_15, ret_60
      Momentum (3):       rsi_14, atr_pct, macd_hist_norm
      Trend / mean-rev (4): bb_pos, sma20_dist, sma50_dist, vol_delta_20
      Swing levels (2):   high_20_dist, low_20_dist
    Total: 20 features
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Bar structure ────────────────────────────────────────────────────────
    for col in ["open", "high", "low", "close"]:
        out[f"ret_{col}"] = df[col].pct_change()

    out["range_pct"] = (high - low) / close

    bar_range = high - low
    bar_range_safe = bar_range.replace(0, np.nan)
    out["upper_wick"] = (high - df[["open", "close"]].max(axis=1)) / bar_range_safe
    out["lower_wick"] = (df[["open", "close"]].min(axis=1) - low) / bar_range_safe
    out["upper_wick"] = out["upper_wick"].fillna(0)
    out["lower_wick"] = out["lower_wick"].fillna(0)

    vol_mean = volume.rolling(vol_window, min_periods=1).mean()
    vol_std = volume.rolling(vol_window, min_periods=1).std().replace(0, 1)
    out["vol_zscore"] = (volume - vol_mean) / vol_std

    # ── Multi-timeframe returns ──────────────────────────────────────────────
    out["ret_5"]  = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)
    out["ret_60"] = close.pct_change(60)

    # ── RSI(14) normalized to [-0.5, 0.5] ───────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_14"] = (rsi.fillna(50) - 50) / 100  # centred at 0, range [-0.5, 0.5]

    # ── ATR(14) as pct of close ──────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    out["atr_pct"] = atr14 / close

    # ── MACD histogram normalised by ATR ────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    out["macd_hist_norm"] = macd_hist / atr14.replace(0, np.nan)

    # ── Bollinger Band position ──────────────────────────────────────────────
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().replace(0, np.nan)
    out["bb_pos"] = (close - sma20) / (2 * std20)  # ~[-1, 1]; >1 = overbought

    # ── Distance from moving averages ───────────────────────────────────────
    out["sma20_dist"] = (close - sma20) / close
    sma50 = close.rolling(50, min_periods=1).mean()
    out["sma50_dist"] = (close - sma50) / close

    # ── Volume delta (fraction of up-bars in last 20) ───────────────────────
    up_bar = (df["close"] > df["open"]).astype(float)
    out["vol_delta_20"] = up_bar.rolling(20, min_periods=1).mean() - 0.5  # centre at 0

    # ── Distance from 20-bar high / low ─────────────────────────────────────
    high_20 = high.rolling(20, min_periods=1).max()
    low_20  = low.rolling(20, min_periods=1).min()
    out["high_20_dist"] = (high_20 - close) / close  # how far below recent high
    out["low_20_dist"]  = (close - low_20)  / close  # how far above recent low

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
    tp_points: float = 30.0,
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
