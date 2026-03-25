"""Realistic backtester with bar-by-bar trade simulation, P&L tracking, and equity curve."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .model import NeuralOHLCVNet
from .preprocessing import compute_features, add_time_features
from .risk import RiskManager, RiskConfig
from .gpu import detect_gpu


@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_bar: int
    exit_bar: int
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    size: int               # contracts
    pnl: float              # dollar P&L
    exit_reason: str         # "TP", "SL", "TIMEOUT"
    confidence: float
    entry_time: pd.Timestamp = None
    exit_time: pd.Timestamp = None


@dataclass
class BacktestResult:
    """Full backtest output."""
    trades: list[Trade]
    equity_curve: pd.Series       # indexed by bar timestamp
    daily_pnl: pd.Series          # daily P&L series
    starting_equity: float
    final_equity: float

    # Summary stats
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    total_pnl: float = 0.0
    avg_rr_realized: float = 0.0
    longest_drawdown_bars: int = 0

    # Consistency metrics (prop firm requirements)
    days_traded: int = 0
    total_trading_days: int = 0
    trading_day_pct: float = 0.0        # % of days with at least 1 trade
    avg_trades_per_day: float = 0.0
    avg_daily_pnl: float = 0.0
    worst_day: float = 0.0
    best_day: float = 0.0
    profitable_days: int = 0
    profitable_day_pct: float = 0.0


class Backtester:
    """Bar-by-bar backtester that simulates trades with actual P&L.

    Loads a trained model, walks through OHLCV data, generates signals,
    applies risk management, and simulates TP/SL resolution using high/low prices.
    """

    def __init__(
        self,
        model_path: str = "model.pt",
        point_value: float = 20.0,       # NQ = $20/point
        starting_equity: float = 50_000.0,
        commission_per_contract: float = 4.50,  # round-trip
        max_bars_in_trade: int = 20,
        risk_config: Optional[RiskConfig] = None,
        lookback: int = 90,
        fixed_tp_points: float = 35.0,   # fixed take-profit in points
        fixed_sl_points: float = 20.0,   # fixed stop-loss in points
    ):
        self.point_value = point_value
        self.starting_equity = starting_equity
        self.commission = commission_per_contract
        self.max_bars_in_trade = max_bars_in_trade
        self.lookback = lookback
        self.fixed_tp_points = fixed_tp_points
        self.fixed_sl_points = fixed_sl_points

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

        # Risk manager
        self.risk_config = risk_config or RiskConfig()
        self.risk_mgr = RiskManager(self.risk_config, starting_equity=starting_equity)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and standardize features using training stats."""
        features = compute_features(df)
        features = add_time_features(features)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        features = (features - self.feat_mean) / self.feat_std.replace(0, 1)
        features = features.clip(-10, 10)
        return features

    @torch.no_grad()
    def _predict(self, window: np.ndarray) -> dict:
        """Run model inference on a single lookback window."""
        X = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
            dir_logits, conf, mag, pred_tp, pred_sl = self.model(X)

        probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
        direction = int(dir_logits.argmax(dim=1).cpu().item())
        return {
            "direction": direction,
            "confidence": float(torch.sigmoid(conf).cpu().item()),
            "magnitude": float(mag.cpu().item()),
            "tp_pct": float(pred_tp.cpu().item()),
            "sl_pct": float(pred_sl.cpu().item()),
            "probs": probs,
        }

    @torch.no_grad()
    def _predict_all(self, feat_values: np.ndarray, bar_indices: list[int]) -> dict[int, dict]:
        """Batch inference on all valid bars at once. Returns {bar_idx: prediction_dict}."""
        if not bar_indices:
            return {}

        batch_size = 512
        results = {}
        total = len(bar_indices)
        print(f"  Batched inference on {total} bars...", end=" ", flush=True)

        import time
        t0 = time.time()

        for batch_start in range(0, total, batch_size):
            batch_idx = bar_indices[batch_start:batch_start + batch_size]
            windows = np.stack([
                feat_values[i - self.lookback + 1:i + 1]
                for i in batch_idx
            ])
            X = torch.from_numpy(windows).to(self.device)

            with torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
                dir_logits, conf, mag, pred_tp, pred_sl = self.model(X)

            probs = torch.softmax(dir_logits, dim=1).cpu().numpy()
            directions = dir_logits.argmax(dim=1).cpu().numpy()
            confidences = torch.sigmoid(conf).cpu().numpy().flatten()
            magnitudes = mag.cpu().numpy().flatten()
            tp_pcts = pred_tp.cpu().numpy().flatten()
            sl_pcts = pred_sl.cpu().numpy().flatten()

            for j, idx in enumerate(batch_idx):
                results[idx] = {
                    "direction": int(directions[j]),
                    "confidence": float(confidences[j]),
                    "magnitude": float(magnitudes[j]),
                    "tp_pct": float(tp_pcts[j]),
                    "sl_pct": float(sl_pcts[j]),
                    "probs": probs[j],
                }

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        return results

    def _resolve_trade(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        index: pd.Index,
        entry_idx: int,
        direction: str,
        entry_price: float,
        size: int,
        confidence: float,
    ) -> Trade:
        """Simulate a trade forward from entry bar using fixed point TP/SL.

        Uses pre-extracted numpy arrays for speed (not df.iloc).
        """
        if direction == "LONG":
            tp_price = entry_price + self.fixed_tp_points
            sl_price = entry_price - self.fixed_sl_points
        else:
            tp_price = entry_price - self.fixed_tp_points
            sl_price = entry_price + self.fixed_sl_points

        exit_bar = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"

        n = len(highs)
        for j in range(1, min(self.max_bars_in_trade + 1, n - entry_idx)):
            bi = entry_idx + j
            bar_high = highs[bi]
            bar_low = lows[bi]
            bar_close = closes[bi]

            if direction == "LONG":
                if bar_low <= sl_price:
                    exit_bar = bi
                    exit_price = sl_price
                    exit_reason = "SL"
                    break
                if bar_high >= tp_price:
                    exit_bar = bi
                    exit_price = tp_price
                    exit_reason = "TP"
                    break
            else:
                if bar_high >= sl_price:
                    exit_bar = bi
                    exit_price = sl_price
                    exit_reason = "SL"
                    break
                if bar_low <= tp_price:
                    exit_bar = bi
                    exit_price = tp_price
                    exit_reason = "TP"
                    break

            if j == self.max_bars_in_trade:
                exit_bar = bi
                exit_price = bar_close
                break

        if exit_bar == entry_idx:
            exit_bar = min(entry_idx + 1, n - 1)
            exit_price = closes[exit_bar]

        if direction == "LONG":
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        pnl = price_diff * self.point_value * size - self.commission * size

        entry_time = index[entry_idx] if isinstance(index, pd.DatetimeIndex) else None
        exit_time = index[exit_bar] if isinstance(index, pd.DatetimeIndex) else None

        return Trade(
            entry_bar=entry_idx,
            exit_bar=exit_bar,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            tp_price=tp_price,
            sl_price=sl_price,
            size=size,
            pnl=pnl,
            exit_reason=exit_reason,
            confidence=confidence,
            entry_time=entry_time,
            exit_time=exit_time,
        )

    def run(
        self,
        df: pd.DataFrame,
        test_pct: float = 0.2,
    ) -> BacktestResult:
        """Run full backtest on the test portion of data.

        Args:
            df: Raw OHLCV dataframe with DatetimeIndex
            test_pct: fraction of data to use as test set (from the end)

        Returns:
            BacktestResult with trades, equity curve, and statistics
        """
        features = self._prepare_features(df)

        # Use last test_pct of data for backtesting (matching training split)
        total_bars = len(features)
        test_start = int(total_bars * (1 - test_pct))

        # Need lookback bars before test_start for the first prediction
        if test_start < self.lookback:
            raise ValueError(f"Not enough data: test starts at {test_start} but need {self.lookback} lookback bars")

        feat_values = features.values.astype(np.float32)

        print(f"\nBacktest range: bar {test_start} to {total_bars}")
        print(f"  Test bars: {total_bars - test_start}")
        print(f"  Date range: {df.index[test_start]} -> {df.index[-1]}")
        print(f"  Starting equity: ${self.starting_equity:,.2f}")
        print(f"  Point value: ${self.point_value}")
        print(f"  Fixed TP: {self.fixed_tp_points} pts (${self.fixed_tp_points * self.point_value})")
        print(f"  Fixed SL: {self.fixed_sl_points} pts (${self.fixed_sl_points * self.point_value})")
        print(f"  R:R: {self.fixed_tp_points / self.fixed_sl_points:.2f}")
        print(f"  Commission: ${self.commission}/contract round-trip")
        print()

        # Pre-extract numpy arrays — avoids df.iloc in inner loops (huge speedup)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        index = df.index

        # Pre-compute time-of-day filter mask
        is_datetime = isinstance(index, pd.DatetimeIndex)
        time_mask = np.ones(total_bars, dtype=bool)
        if is_datetime:
            hour_mins = index.hour * 60 + index.minute
            time_mask = ~(((hour_mins >= 810) & (hour_mins <= 840)) |
                          ((hour_mins >= 1170) & (hour_mins <= 1200)))

        # Collect all valid bar indices for batched inference
        valid_bars = []
        for i in range(test_start, total_bars - 1):
            if i < self.lookback:
                continue
            if not time_mask[i]:
                continue
            valid_bars.append(i)

        # Batch inference — run model on ALL valid bars at once
        predictions = self._predict_all(feat_values, valid_bars)

        # Now walk through bars with cached predictions (no more GPU calls)
        trades: list[Trade] = []
        equity = self.starting_equity
        equity_series = {}
        in_trade = False
        current_trade_exit_bar = 0
        current_day = None
        skipped = {"flat": 0, "low_conf": 0, "risk_rr": 0, "risk_halted": 0, "risk_cooldown": 0, "max_trades": 0, "bar_gap": 0, "risk_other": 0}

        for i in range(test_start, total_bars - 1):
            ts = index[i]
            if is_datetime:
                day = ts.date()
                if day != current_day:
                    self.risk_mgr.reset_daily()
                    current_day = day

            equity_series[ts] = equity

            if in_trade and i <= current_trade_exit_bar:
                continue
            in_trade = False

            # Skip bars that weren't predicted (time filter / not enough lookback)
            if i not in predictions:
                if not time_mask[i]:
                    skipped["flat"] += 1
                continue

            pred = predictions[i]
            direction_idx = pred["direction"]
            confidence = pred["confidence"]

            if direction_idx == 0:
                direction = "LONG"
            elif direction_idx == 1:
                direction = "SHORT"
            else:
                skipped["flat"] += 1
                continue

            entry_approx = closes[i]
            tp_pct_approx = self.fixed_tp_points / entry_approx
            sl_pct_approx = self.fixed_sl_points / entry_approx

            allowed, size, reason = self.risk_mgr.check_trade(
                confidence, direction_idx, tp_pct_approx, sl_pct_approx,
                current_bar=i,
            )
            if not allowed:
                if "Confidence" in reason:
                    skipped["low_conf"] += 1
                elif "R:R" in reason:
                    skipped["risk_rr"] += 1
                elif "halted" in reason.lower() or "limit" in reason.lower():
                    skipped["risk_halted"] += 1
                elif "Cooldown" in reason:
                    skipped["risk_cooldown"] += 1
                elif "Max trades" in reason:
                    skipped["max_trades"] += 1
                elif "bar gap" in reason:
                    skipped["bar_gap"] += 1
                else:
                    skipped["risk_other"] += 1
                continue

            entry_idx = i + 1
            if entry_idx >= total_bars:
                break
            entry_price = opens[entry_idx]

            trade = self._resolve_trade(
                highs, lows, closes, index,
                entry_idx, direction, entry_price,
                size, confidence,
            )
            trades.append(trade)

            # Update state
            equity += trade.pnl
            self.risk_mgr.record_trade_result(trade.pnl, entry_bar=i, confidence=trade.confidence)
            in_trade = True
            current_trade_exit_bar = trade.exit_bar

            # Check if halted
            if self.risk_mgr.state.is_halted:
                print(f"  HALTED at bar {i}: {self.risk_mgr.state.halt_reason}")

        # Fill remaining equity
        for i in range(max(test_start, current_trade_exit_bar + 1), total_bars):
            equity_series[df.index[i]] = equity

        equity_curve = pd.Series(equity_series).sort_index()

        print(f"  Signals skipped: flat={skipped['flat']} low_conf={skipped['low_conf']} "
              f"rr={skipped['risk_rr']} halted={skipped['risk_halted']} cooldown={skipped['risk_cooldown']} "
              f"max_trades={skipped['max_trades']} bar_gap={skipped['bar_gap']} other={skipped['risk_other']}")

        result = self._compute_stats(trades, equity_curve, equity)
        self._print_report(result)
        return result

    def _compute_stats(
        self,
        trades: list[Trade],
        equity_curve: pd.Series,
        final_equity: float,
    ) -> BacktestResult:
        """Compute comprehensive backtest statistics."""
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            daily_pnl=pd.Series(dtype=float),
            starting_equity=self.starting_equity,
            final_equity=final_equity,
        )

        if not trades:
            return result

        pnls = np.array([t.pnl for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        result.total_trades = len(trades)
        result.winners = sum(1 for t in trades if t.exit_reason == "TP")
        result.losers = sum(1 for t in trades if t.exit_reason == "SL")
        result.timeouts = sum(1 for t in trades if t.exit_reason == "TIMEOUT")
        result.win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        result.avg_win = float(wins.mean()) if len(wins) > 0 else 0
        result.avg_loss = float(losses.mean()) if len(losses) > 0 else 0
        result.total_pnl = float(pnls.sum())

        gross_wins = float(wins.sum()) if len(wins) > 0 else 0
        gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 1
        result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Realized R:R
        if result.avg_loss != 0:
            result.avg_rr_realized = abs(result.avg_win / result.avg_loss)

        # Drawdown from equity curve
        peak = equity_curve.cummax()
        drawdown = equity_curve - peak
        result.max_drawdown = float(drawdown.min())
        result.max_drawdown_pct = float((drawdown / peak).min()) * 100

        # Longest drawdown duration (in bars)
        in_dd = drawdown < 0
        if in_dd.any():
            groups = (~in_dd).cumsum()
            dd_lengths = in_dd.groupby(groups).sum()
            result.longest_drawdown_bars = int(dd_lengths.max())

        # Daily P&L
        if isinstance(equity_curve.index, pd.DatetimeIndex):
            daily_equity = equity_curve.resample("D").last().dropna()
            result.daily_pnl = daily_equity.diff().dropna()

            # Sharpe (annualized from daily)
            if len(result.daily_pnl) > 1:
                daily_ret = result.daily_pnl / self.starting_equity
                if daily_ret.std() > 0:
                    result.sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

                # Sortino (annualized from daily)
                downside = daily_ret[daily_ret < 0]
                if len(downside) > 0 and downside.std() > 0:
                    result.sortino = float(daily_ret.mean() / downside.std() * np.sqrt(252))

        # Consistency metrics
        if trades and trades[0].entry_time is not None:
            trade_days = set()
            for t in trades:
                if t.entry_time is not None:
                    trade_days.add(t.entry_time.date())

            if isinstance(equity_curve.index, pd.DatetimeIndex):
                all_days = set(equity_curve.index.date)
                result.total_trading_days = len(all_days)
                result.days_traded = len(trade_days)
                result.trading_day_pct = result.days_traded / max(result.total_trading_days, 1) * 100
                result.avg_trades_per_day = result.total_trades / max(result.days_traded, 1)

            if len(result.daily_pnl) > 0:
                result.avg_daily_pnl = float(result.daily_pnl.mean())
                result.worst_day = float(result.daily_pnl.min())
                result.best_day = float(result.daily_pnl.max())
                result.profitable_days = int((result.daily_pnl > 0).sum())
                days_with_trades = len(result.daily_pnl[result.daily_pnl != 0])
                result.profitable_day_pct = result.profitable_days / max(days_with_trades, 1) * 100

        return result

    def _print_report(self, r: BacktestResult) -> None:
        """Print a formatted backtest report."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST REPORT")
        print(f"{'='*60}")
        print(f"  Total trades:      {r.total_trades}")
        print(f"  Winners (TP):      {r.winners}  |  Losers (SL): {r.losers}  |  Timeouts: {r.timeouts}")
        print(f"  Win rate:          {r.win_rate:.1%}")
        print(f"  Avg win:           ${r.avg_win:,.2f}")
        print(f"  Avg loss:          ${r.avg_loss:,.2f}")
        print(f"  Avg R:R realized:  {r.avg_rr_realized:.2f}")
        print(f"  Profit factor:     {r.profit_factor:.2f}")
        print()
        print(f"  Starting equity:   ${r.starting_equity:,.2f}")
        print(f"  Final equity:      ${r.final_equity:,.2f}")
        print(f"  Total P&L:         ${r.total_pnl:,.2f}")
        print(f"  Max drawdown:      ${r.max_drawdown:,.2f} ({r.max_drawdown_pct:.2f}%)")
        print(f"  Longest DD:        {r.longest_drawdown_bars} bars")
        print()
        print(f"  Sharpe (annual):   {r.sharpe:.2f}")
        print(f"  Sortino (annual):  {r.sortino:.2f}")
        print()
        print(f"  CONSISTENCY (Prop Firm)")
        print(f"  Days traded:       {r.days_traded} / {r.total_trading_days} ({r.trading_day_pct:.0f}%)")
        print(f"  Avg trades/day:    {r.avg_trades_per_day:.1f}")
        print(f"  Profitable days:   {r.profitable_days} ({r.profitable_day_pct:.0f}%)")
        print(f"  Avg daily P&L:     ${r.avg_daily_pnl:,.2f}")
        print(f"  Best day:          ${r.best_day:,.2f}")
        print(f"  Worst day:         ${r.worst_day:,.2f}")
        print(f"{'='*60}")

    def plot_equity_curve(self, result: BacktestResult, save_path: str = "equity_curve.png") -> None:
        """Plot and save equity curve with drawdown overlay."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("  matplotlib not installed — skipping plot")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                              gridspec_kw={"height_ratios": [3, 1, 1]})

        # Equity curve
        eq = result.equity_curve
        ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=1.2, label="Equity")
        ax1.axhline(result.starting_equity, color="gray", linestyle="--", alpha=0.5, label="Starting")
        ax1.fill_between(eq.index, result.starting_equity, eq.values,
                         where=eq.values >= result.starting_equity, alpha=0.15, color="green")
        ax1.fill_between(eq.index, result.starting_equity, eq.values,
                         where=eq.values < result.starting_equity, alpha=0.15, color="red")
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(f"Backtest Equity Curve  |  P&L: ${result.total_pnl:,.0f}  |  "
                       f"Sharpe: {result.sharpe:.2f}  |  Sortino: {result.sortino:.2f}")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Trade markers
        for trade in result.trades:
            if trade.entry_time is not None:
                color = "green" if trade.pnl > 0 else "red"
                marker = "^" if trade.direction == "LONG" else "v"
                # Find equity at entry time
                if trade.entry_time in eq.index:
                    y = eq[trade.entry_time]
                else:
                    y = eq.iloc[eq.index.get_indexer([trade.entry_time], method="nearest")[0]]
                ax1.scatter(trade.entry_time, y, color=color, marker=marker, s=20, alpha=0.7, zorder=5)

        # Drawdown
        peak = eq.cummax()
        dd = eq - peak
        ax2.fill_between(eq.index, 0, dd.values, color="red", alpha=0.4)
        ax2.set_ylabel("Drawdown ($)")
        ax2.grid(True, alpha=0.3)

        # Per-trade P&L
        if result.trades:
            trade_times = [t.exit_time or eq.index[min(t.exit_bar, len(eq) - 1)] for t in result.trades]
            trade_pnls = [t.pnl for t in result.trades]
            colors = ["green" if p > 0 else "red" for p in trade_pnls]
            ax3.bar(trade_times, trade_pnls, color=colors, alpha=0.7, width=0.01)
            ax3.axhline(0, color="gray", linewidth=0.5)
            ax3.set_ylabel("Trade P&L ($)")
            ax3.grid(True, alpha=0.3)

        if isinstance(eq.index, pd.DatetimeIndex):
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Equity curve saved to {save_path}")

    def export_trades(self, result: BacktestResult, save_path: str = "trades.csv") -> None:
        """Export trade log to CSV."""
        if not result.trades:
            print("  No trades to export")
            return

        rows = []
        for t in result.trades:
            rows.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "tp_price": t.tp_price,
                "sl_price": t.sl_price,
                "size": t.size,
                "pnl": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
                "confidence": round(t.confidence, 3),
            })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"  Trade log saved to {save_path} ({len(rows)} trades)")
