"""Grid search optimizer — run inference once, replay trade logic across all param combos."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time
import sys

from numba import njit

from .model import NeuralOHLCVNet
from .preprocessing import compute_features, add_time_features
from .gpu import detect_gpu


@njit(cache=True)
def _simulate_combo_jit(
    signal_bars: np.ndarray,      # int64 — bar indices
    signal_dirs: np.ndarray,      # int64 — 0=LONG, 1=SHORT
    signal_confs: np.ndarray,     # float64 — confidences
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    sl_pts: float,
    tp_pts: float,
    min_conf: float,
    max_bars: int,
    point_value: float,
    commission: float,
    starting_equity: float,
) -> tuple:
    """Pure-numeric trade simulation, JIT-compiled by Numba.

    Returns (total_trades, winners, losers, timeouts,
             total_pnl, total_win_pnl, total_loss_pnl,
             max_dd, max_dd_pct).
    """
    equity = starting_equity
    peak_equity = equity
    max_dd = 0.0
    max_dd_pct = 0.0

    winners = 0
    losers = 0
    timeouts = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    total_pnl = 0.0

    in_trade = False
    trade_exit_bar = 0
    n = len(closes)

    for s in range(len(signal_bars)):
        bar_idx = signal_bars[s]
        direction = signal_dirs[s]
        confidence = signal_confs[s]

        if confidence < min_conf:
            continue

        if in_trade and bar_idx <= trade_exit_bar:
            continue
        in_trade = False

        entry_idx = bar_idx + 1
        if entry_idx >= n:
            break
        entry_price = opens[entry_idx]

        if direction == 0:  # LONG
            tp_price = entry_price + tp_pts
            sl_price = entry_price - sl_pts
        else:  # SHORT
            tp_price = entry_price - tp_pts
            sl_price = entry_price + sl_pts

        exit_reason = 0  # 0=TIMEOUT, 1=TP, 2=SL
        exit_price = entry_price
        exit_bar = entry_idx

        max_j = min(max_bars + 1, n - entry_idx)
        for j in range(1, max_j):
            bi = entry_idx + j
            bh = highs[bi]
            bl = lows[bi]
            bc = closes[bi]

            if direction == 0:  # LONG
                if bl <= sl_price:
                    exit_bar = bi
                    exit_price = sl_price
                    exit_reason = 2
                    break
                if bh >= tp_price:
                    exit_bar = bi
                    exit_price = tp_price
                    exit_reason = 1
                    break
            else:  # SHORT
                if bh >= sl_price:
                    exit_bar = bi
                    exit_price = sl_price
                    exit_reason = 2
                    break
                if bl <= tp_price:
                    exit_bar = bi
                    exit_price = tp_price
                    exit_reason = 1
                    break

            if j == max_bars:
                exit_bar = bi
                exit_price = bc
                break

        if exit_bar == entry_idx:
            exit_bar = min(entry_idx + 1, n - 1)
            exit_price = closes[exit_bar]

        if direction == 0:
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        pnl = price_diff * point_value - commission

        if exit_reason == 1:  # TP
            winners += 1
            total_win_pnl += pnl
        elif exit_reason == 2:  # SL
            losers += 1
            total_loss_pnl += pnl
        else:
            timeouts += 1
            if pnl > 0:
                total_win_pnl += pnl
            else:
                total_loss_pnl += pnl

        total_pnl += pnl
        equity += pnl

        if equity > peak_equity:
            peak_equity = equity
        dd = equity - peak_equity
        if dd < max_dd:
            max_dd = dd
            if peak_equity > 0:
                max_dd_pct = dd / peak_equity * 100

        in_trade = True
        trade_exit_bar = exit_bar

    total_trades = winners + losers + timeouts
    return (total_trades, winners, losers, timeouts,
            total_pnl, total_win_pnl, total_loss_pnl,
            max_dd, max_dd_pct)


@dataclass
class ComboResult:
    """Stats for a single TP/SL/confidence combo."""
    sl: float
    tp: float
    min_conf: float
    total_trades: int
    winners: int
    losers: int
    timeouts: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_rr: float


class GridOptimizer:
    """Run model inference once, then grid-search TP/SL/confidence params."""

    def __init__(
        self,
        model_path: str = "model.pt",
        point_value: float = 20.0,
        starting_equity: float = 50_000.0,
        commission: float = 4.50,
        max_bars_in_trade: int = 20,
        lookback: int = 90,
        test_pct: float = 0.2,
    ):
        self.point_value = point_value
        self.starting_equity = starting_equity
        self.commission = commission
        self.max_bars = max_bars_in_trade
        self.lookback = lookback
        self.test_pct = test_pct

        self.gpu = detect_gpu()
        self.device = self.gpu.device

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.feat_mean = pd.Series(checkpoint["feat_mean"])
        self.feat_std = pd.Series(checkpoint["feat_std"])

        self.model = NeuralOHLCVNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        state = checkpoint["model_state"]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = compute_features(df)
        features = add_time_features(features)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        features = (features - self.feat_mean) / self.feat_std.replace(0, 1)
        features = features.clip(-10, 10)
        return features

    @torch.no_grad()
    def _run_all_predictions(self, feat_values: np.ndarray, test_start: int) -> list:
        """Run inference on every valid bar in the test set. Returns list of (bar_idx, direction, confidence)."""
        signals = []
        total = len(feat_values)
        batch_size = 256

        # Build all windows first
        indices = []
        for i in range(test_start, total - 1):
            if i < self.lookback:
                continue
            indices.append(i)

        print(f"  Running inference on {len(indices)} bars...", end=" ", flush=True)
        t0 = time.time()

        # Batch inference
        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            windows = np.stack([
                feat_values[i - self.lookback + 1:i + 1]
                for i in batch_idx
            ])
            X = torch.from_numpy(windows.astype(np.float32)).to(self.device)

            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, mag, pred_tp, pred_sl = self.model(X)

            probs = torch.softmax(dir_logits, dim=1)
            directions = dir_logits.argmax(dim=1).cpu().numpy()
            confidences = torch.sigmoid(conf).cpu().numpy().flatten()

            for j, idx in enumerate(batch_idx):
                d = int(directions[j])
                if d == 2:  # FLAT — skip
                    continue
                signals.append((idx, d, float(confidences[j])))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s, {len(signals)} non-flat signals)")

        # Convert to structured numpy arrays for Numba
        if signals:
            signal_bars = np.array([s[0] for s in signals], dtype=np.int64)
            signal_dirs = np.array([s[1] for s in signals], dtype=np.int64)
            signal_confs = np.array([s[2] for s in signals], dtype=np.float64)
        else:
            signal_bars = np.empty(0, dtype=np.int64)
            signal_dirs = np.empty(0, dtype=np.int64)
            signal_confs = np.empty(0, dtype=np.float64)

        return signal_bars, signal_dirs, signal_confs

    def _simulate_combo(
        self,
        signal_bars: np.ndarray,
        signal_dirs: np.ndarray,
        signal_confs: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        sl_pts: float,
        tp_pts: float,
        min_conf: float,
    ) -> ComboResult:
        """Replay trade logic via JIT-compiled function."""
        (total_trades, winners, losers, timeouts,
         total_pnl, total_win_pnl, total_loss_pnl,
         max_dd, max_dd_pct) = _simulate_combo_jit(
            signal_bars, signal_dirs, signal_confs,
            highs, lows, closes, opens,
            sl_pts, tp_pts, min_conf,
            self.max_bars, self.point_value, self.commission,
            self.starting_equity,
        )

        avg_win = total_win_pnl / max(winners, 1)
        avg_loss = total_loss_pnl / max(losers, 1)
        pf = total_win_pnl / max(abs(total_loss_pnl), 1.0)
        wr = winners / max(total_trades, 1)
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        return ComboResult(
            sl=sl_pts, tp=tp_pts, min_conf=min_conf,
            total_trades=total_trades, winners=winners, losers=losers, timeouts=timeouts,
            win_rate=wr, profit_factor=pf, total_pnl=total_pnl,
            avg_win=avg_win, avg_loss=avg_loss,
            max_drawdown=max_dd, max_drawdown_pct=max_dd_pct,
            avg_rr=avg_rr,
        )

    def run(
        self,
        df: pd.DataFrame,
        sl_range: list[float],
        tp_range: list[float],
        conf_range: list[float],
    ) -> pd.DataFrame:
        """Run full grid search. Returns DataFrame of all combo results sorted by total P&L."""
        features = self._prepare_features(df)
        total_bars = len(features)
        test_start = int(total_bars * (1 - self.test_pct))

        if test_start < self.lookback:
            raise ValueError(f"Not enough data")

        feat_values = features.values.astype(np.float32)

        print(f"\nGrid Optimizer")
        print(f"  Data: {total_bars} bars, test from bar {test_start}")
        print(f"  Date range: {df.index[test_start]} -> {df.index[-1]}")
        print(f"  SL range: {sl_range[0]}-{sl_range[-1]} pts ({len(sl_range)} values)")
        print(f"  TP range: {tp_range[0]}-{tp_range[-1]} pts ({len(tp_range)} values)")
        print(f"  Conf range: {conf_range[0]:.0%}-{conf_range[-1]:.0%} ({len(conf_range)} values)")
        print(f"  Total combos: {len(sl_range) * len(tp_range) * len(conf_range):,}")
        print()

        # Step 1: Run inference ONCE
        signal_bars, signal_dirs, signal_confs = self._run_all_predictions(feat_values, test_start)

        # Pre-extract price arrays as contiguous float64 for Numba
        highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
        lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
        closes = np.ascontiguousarray(df["close"].values, dtype=np.float64)
        opens = np.ascontiguousarray(df["open"].values, dtype=np.float64)

        # Warm up Numba JIT (first call compiles; subsequent calls are native speed)
        print("  Compiling JIT...", end=" ", flush=True)
        t_jit = time.time()
        _simulate_combo_jit(
            signal_bars[:1], signal_dirs[:1], signal_confs[:1],
            highs, lows, closes, opens,
            sl_range[0], tp_range[0], conf_range[0],
            self.max_bars, self.point_value, self.commission,
            self.starting_equity,
        )
        print(f"done ({time.time() - t_jit:.1f}s)")

        # Step 2: Sweep all combos
        total_combos = len(sl_range) * len(tp_range) * len(conf_range)
        results = []
        t0 = time.time()
        done = 0

        for sl in sl_range:
            for tp in tp_range:
                for conf in conf_range:
                    r = self._simulate_combo(
                        signal_bars, signal_dirs, signal_confs,
                        highs, lows, closes, opens, sl, tp, conf,
                    )
                    results.append(r)
                    done += 1

                # Progress update per TP row
                elapsed = time.time() - t0
                pct = done / total_combos * 100
                rate = done / max(elapsed, 0.001)
                eta = (total_combos - done) / max(rate, 0.001)
                sys.stdout.write(
                    f"\r  Progress: {done:,}/{total_combos:,} ({pct:.0f}%) "
                    f"| {rate:.0f} combos/s | ETA {eta:.0f}s"
                )
                sys.stdout.flush()

        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s ({total_combos / max(elapsed, 0.001):.0f} combos/s)\n")

        # Build results DataFrame
        rows = []
        for r in results:
            rows.append({
                "SL": r.sl,
                "TP": r.tp,
                "MinConf": r.min_conf,
                "Trades": r.total_trades,
                "Winners": r.winners,
                "Losers": r.losers,
                "Timeouts": r.timeouts,
                "WinRate": r.win_rate,
                "PF": r.profit_factor,
                "TotalPnL": r.total_pnl,
                "AvgWin": r.avg_win,
                "AvgLoss": r.avg_loss,
                "AvgRR": r.avg_rr,
                "MaxDD": r.max_drawdown,
                "MaxDD%": r.max_drawdown_pct,
                "RR_ratio": r.tp / r.sl,
            })

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values("TotalPnL", ascending=False).reset_index(drop=True)
        return results_df

    def print_top(self, results_df: pd.DataFrame, n: int = 25) -> None:
        """Print the top N combos."""
        print(f"{'='*110}")
        print(f"  TOP {n} SETUPS BY TOTAL P&L")
        print(f"{'='*110}")

        header = (
            f"{'#':>3s} {'SL':>5s} {'TP':>5s} {'R:R':>5s} {'Conf':>5s} "
            f"{'Trades':>6s} {'WR':>6s} {'PF':>6s} "
            f"{'Total P&L':>11s} {'AvgWin':>8s} {'AvgLoss':>8s} "
            f"{'MaxDD':>9s} {'DD%':>6s}"
        )
        print(header)
        print("-" * len(header))

        for i, row in results_df.head(n).iterrows():
            line = (
                f"{i+1:>3d} {row['SL']:>5.0f} {row['TP']:>5.0f} {row['RR_ratio']:>5.2f} "
                f"{row['MinConf']:>4.0%} "
                f"{row['Trades']:>6.0f} {row['WinRate']:>5.1%} {row['PF']:>6.2f} "
                f"${row['TotalPnL']:>+9,.0f} ${row['AvgWin']:>7,.0f} ${row['AvgLoss']:>7,.0f} "
                f"${row['MaxDD']:>8,.0f} {row['MaxDD%']:>5.1f}%"
            )
            print(line)

        print(f"{'='*110}")

        # Also show top by Sharpe-like metric (PnL / |MaxDD|) as a simple risk-adjusted rank
        results_df = results_df.copy()
        results_df["PnL_DD_ratio"] = results_df["TotalPnL"] / results_df["MaxDD"].abs().clip(lower=1)
        top_risk = results_df.sort_values("PnL_DD_ratio", ascending=False).head(n)

        print(f"\n{'='*110}")
        print(f"  TOP {n} SETUPS BY P&L / DRAWDOWN RATIO (risk-adjusted)")
        print(f"{'='*110}")
        print(header)
        print("-" * len(header))

        for rank, (i, row) in enumerate(top_risk.iterrows()):
            line = (
                f"{rank+1:>3d} {row['SL']:>5.0f} {row['TP']:>5.0f} {row['RR_ratio']:>5.2f} "
                f"{row['MinConf']:>4.0%} "
                f"{row['Trades']:>6.0f} {row['WinRate']:>5.1%} {row['PF']:>6.2f} "
                f"${row['TotalPnL']:>+9,.0f} ${row['AvgWin']:>7,.0f} ${row['AvgLoss']:>7,.0f} "
                f"${row['MaxDD']:>8,.0f} {row['MaxDD%']:>5.1f}%"
            )
            print(line)

        print(f"{'='*110}")

        # Filter: only show combos with at least 30 trades
        min_trades = 30
        filtered = results_df[results_df["Trades"] >= min_trades]
        if len(filtered) > 0:
            top_filtered = filtered.sort_values("PnL_DD_ratio", ascending=False).head(n)
            print(f"\n{'='*110}")
            print(f"  TOP {n} (min {min_trades} trades) BY P&L / DRAWDOWN RATIO")
            print(f"{'='*110}")
            print(header)
            print("-" * len(header))

            for rank, (i, row) in enumerate(top_filtered.iterrows()):
                line = (
                    f"{rank+1:>3d} {row['SL']:>5.0f} {row['TP']:>5.0f} {row['RR_ratio']:>5.2f} "
                    f"{row['MinConf']:>4.0%} "
                    f"{row['Trades']:>6.0f} {row['WinRate']:>5.1%} {row['PF']:>6.2f} "
                    f"${row['TotalPnL']:>+9,.0f} ${row['AvgWin']:>7,.0f} ${row['AvgLoss']:>7,.0f} "
                    f"${row['MaxDD']:>8,.0f} {row['MaxDD%']:>5.1f}%"
                )
                print(line)

            print(f"{'='*110}")
