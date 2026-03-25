"""Live trading module — streams OHLCV bars from Databento and runs model inference."""

import os
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch

import databento as db

from .model import NeuralOHLCVNet
from .config import Config
from .risk import RiskManager, RiskConfig
from .preprocessing import compute_features, add_time_features
from .gpu import detect_gpu


class BarBuffer:
    """Rolling buffer of OHLCV bars for feature computation.

    Keeps enough history to compute features (vol_window) and build
    a full lookback sequence for the model.
    """

    def __init__(self, lookback: int = 90, vol_window: int = 20):
        self.lookback = lookback
        self.vol_window = vol_window
        # Need extra bars for pct_change (1) and vol rolling window
        self.min_bars = lookback + vol_window + 2
        self._bars: list[dict] = []

    def add_bar(self, ts: datetime, o: float, h: float, l: float, c: float, v: float) -> None:
        self._bars.append({
            "timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
        # Keep a bounded buffer (2x what we need)
        max_keep = self.min_bars * 2
        if len(self._bars) > max_keep:
            self._bars = self._bars[-max_keep:]

    @property
    def ready(self) -> bool:
        return len(self._bars) >= self.min_bars

    @property
    def bar_count(self) -> int:
        return len(self._bars)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._bars)
        df.index = pd.DatetimeIndex(df.pop("timestamp"))
        return df

    @property
    def last_close(self) -> float:
        return self._bars[-1]["close"] if self._bars else 0.0

    @property
    def last_timestamp(self) -> Optional[datetime]:
        return self._bars[-1]["timestamp"] if self._bars else None


class LiveTrader:
    """Connects to Databento live feed, runs model on each new bar, and logs signals."""

    PRICE_SCALE = 1e-9  # Databento fixed-point price scale

    def __init__(
        self,
        model_path: str,
        dataset: str = "GLBX.MDP3",
        schema: str = "ohlcv-1m",
        symbols: Optional[list[str]] = None,
        stype_in: str = "continuous",
        api_key: Optional[str] = None,
        paper: bool = True,
    ):
        self.dataset = dataset
        self.schema = schema
        self.symbols = symbols or ["NQ.c.0"]
        self.stype_in = stype_in
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
        self.paper = paper

        if not self.api_key:
            raise ValueError(
                "Databento API key required. Set DATABENTO_API_KEY env var "
                "or pass api_key parameter."
            )

        # Load model
        self.config = Config()
        self.gpu = detect_gpu()
        self._load_model(model_path)

        # Bar buffer
        self.buffer = BarBuffer(
            lookback=self.config.lookback,
            vol_window=self.config.vol_window,
        )

        # Fixed TP/SL from config
        self.fixed_tp_points = self.config.fixed_tp_points
        self.fixed_sl_points = self.config.fixed_sl_points

        # Risk manager
        self.risk_mgr = RiskManager(
            RiskConfig(
                daily_loss_limit=self.config.daily_loss_limit,
                min_confidence=self.config.min_confidence,
                cooldown_bars=self.config.cooldown_bars,
                consecutive_loss_trigger=self.config.consecutive_loss_trigger,
                max_position_size=self.config.max_position_size,
                max_trades_per_day=self.config.max_trades_per_day,
                min_bars_between_trades=self.config.min_bars_between_trades,
            )
        )

        # Trade tracking
        self._trade_log: list[dict] = []
        self._current_position: Optional[dict] = None
        self._bar_count = 0
        self._signal_count = 0
        self._last_day: Optional[int] = None

        # Live CSV files — written incrementally so data survives crashes
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trades_csv = f"live_trades_{session_ts}.csv"
        self._buckets_csv = f"live_buckets_{session_ts}.csv"
        self._trades_csv_header_written = False

    def _load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.gpu.device, weights_only=True)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]

        self.model = NeuralOHLCVNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.gpu.device)

        # Strip _orig_mod. prefix added by torch.compile()
        state = checkpoint["model_state"]
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        # Load normalization stats
        if "feat_mean" in checkpoint and "feat_std" in checkpoint:
            self.feat_mean = pd.Series(checkpoint["feat_mean"])
            self.feat_std = pd.Series(checkpoint["feat_std"])
            print(f"  Loaded normalization stats ({len(self.feat_mean)} features)")
        else:
            self.feat_mean = None
            self.feat_std = None
            print("  WARNING: No normalization stats in checkpoint — using live estimates")

        metrics = checkpoint.get("test_metrics", {})
        print(f"  Model: input_dim={self.input_dim} hidden_dim={self.hidden_dim}")
        if metrics:
            print(f"  Backtest: acc={metrics.get('accuracy', 0):.3f} "
                  f"sortino={metrics.get('sortino', 0):.3f}")

    def _process_bar(self, ts: datetime, o: float, h: float, l: float, c: float, v: float) -> None:
        """Called on each new OHLCV bar."""
        self.buffer.add_bar(ts, o, h, l, c, v)
        self._bar_count += 1

        # Reset risk manager at start of new trading day
        if self._last_day is not None and ts.day != self._last_day:
            self.risk_mgr.reset_daily()
            print(f"\n{'='*60}")
            print(f"  NEW TRADING DAY: {ts.strftime('%Y-%m-%d')}")
            print(f"{'='*60}")
        self._last_day = ts.day

        if not self.buffer.ready:
            pct = self.buffer.bar_count / self.buffer.min_bars * 100
            print(f"  [{ts.strftime('%H:%M:%S')}] Buffering {self.buffer.bar_count}/{self.buffer.min_bars} "
                  f"({pct:.0f}%) | {c:.2f}")
            return

        # Run inference
        signal = self._infer()
        if signal is None:
            print(f"  [{ts.strftime('%H:%M:%S')}] bar #{self._bar_count} | {c:.2f} | no signal")
            return
        self._handle_signal(signal, ts, o, h, l, c)

    def _infer(self) -> Optional[dict]:
        """Run model inference on current buffer."""
        df = self.buffer.to_dataframe()

        # Compute features
        features = compute_features(df, vol_window=self.config.vol_window)
        features = add_time_features(features)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Normalize using saved training stats
        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / self.feat_std
        else:
            # Fallback: normalize from buffer history (less accurate)
            mean = features.mean()
            std = features.std().replace(0, 1)
            features = (features - mean) / std

        features = features.clip(-10, 10)

        # Take last lookback bars
        feat_vals = features.values[-self.config.lookback:].astype(np.float32)
        if len(feat_vals) < self.config.lookback:
            return None

        X = torch.from_numpy(feat_vals).unsqueeze(0).to(self.gpu.device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.gpu.use_amp):
            dir_logits, confidence, magnitude, pred_tp, pred_sl = self.model(X)

        probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
        conf = torch.sigmoid(confidence).cpu().item()
        mag = magnitude.cpu().item()
        direction = int(dir_logits.argmax(dim=1).cpu().item())

        return {
            "direction": direction,
            "probs": probs,
            "confidence": conf,
            "magnitude": mag,
        }

    def _handle_signal(self, signal: dict, ts: datetime, bar_open: float, bar_high: float, bar_low: float, close_price: float) -> None:
        """Process a model signal through risk management and log it."""
        direction = signal["direction"]
        conf = signal["confidence"]
        probs = signal["probs"]
        labels = ["LONG", "SHORT", "FLAT"]
        last_close = self.buffer.last_close

        # Use fixed TP/SL as pct for risk check
        tp_pct = self.fixed_tp_points / last_close if last_close > 0 else 0
        sl_pct = self.fixed_sl_points / last_close if last_close > 0 else 0

        # Check with risk manager
        allowed, size, reason = self.risk_mgr.check_trade(
            conf, direction, tp_pct, sl_pct,
            current_bar=self._signal_count,
        )

        self._signal_count += 1

        # Resolve any open position (simulated) using intrabar high/low
        if self._current_position is not None:
            self._resolve_position(bar_high, bar_low, last_close, ts)

        # Log the signal
        ts_str = ts.strftime("%H:%M:%S")
        dir_label = labels[direction]
        prob_str = f"W={probs[0]:.2f} L={probs[1]:.2f} F={probs[2]:.2f}"

        # Block new entry if a position is still open
        if self._current_position is not None:
            allowed = False
            reason = "position_open"

        if allowed:
            if direction == 0:  # LONG
                tp_price = last_close + self.fixed_tp_points
                sl_price = last_close - self.fixed_sl_points
            else:  # SHORT
                tp_price = last_close - self.fixed_tp_points
                sl_price = last_close + self.fixed_sl_points
            rr = self.fixed_tp_points / self.fixed_sl_points

            print(
                f"  [{ts_str}] #{self._signal_count:4d} "
                f"{dir_label:5s} x{size} @ {last_close:.2f} | "
                f"conf={conf:.2f} ({prob_str}) | "
                f"TP={tp_price:.2f}(+{self.fixed_tp_points:.0f}pt) "
                f"SL={sl_price:.2f}(-{self.fixed_sl_points:.0f}pt) "
                f"R:R={rr:.1f}"
            )

            if self.paper:
                self._current_position = {
                    "direction": direction,
                    "entry_price": last_close,
                    "size": size,
                    "tp_points": self.fixed_tp_points,
                    "sl_points": self.fixed_sl_points,
                    "entry_time": ts,
                    "bars_held": 0,
                    "trade_log_idx": len(self._trade_log),  # index into trade log
                    "confidence": confidence,
                }
        else:
            if direction == 2:
                # FLAT — just show a heartbeat line
                print(
                    f"  [{ts_str}] #{self._signal_count:4d} "
                    f" FLAT  @ {last_close:.2f} | "
                    f"conf={conf:.2f} ({prob_str})"
                )
            else:
                print(
                    f"  [{ts_str}] #{self._signal_count:4d} "
                    f"{dir_label:5s} BLOCKED | {reason} | "
                    f"conf={conf:.2f} ({prob_str})"
                )

        # Log trade for analysis
        entry = {
            "timestamp": ts,
            "direction": dir_label,
            "confidence": conf,
            "probs_win": probs[0],
            "probs_lose": probs[1],
            "probs_flat": probs[2],
            "magnitude": signal["magnitude"],
            "tp_points": self.fixed_tp_points,
            "sl_points": self.fixed_sl_points,
            "price": last_close,
            "allowed": allowed,
            "size": size if allowed else 0,
            "reason": reason,
            "result": None,
            "pnl": None,
        }
        self._trade_log.append(entry)
        self._save_trade_row(entry)

    def _resolve_position(self, bar_high: float, bar_low: float, bar_close: float, ts: datetime) -> None:
        """Simulate resolving the previous position using intrabar prices."""
        pos = self._current_position
        if pos is None:
            return

        pos["bars_held"] += 1
        entry = pos["entry_price"]
        tp_pts = pos["tp_points"]
        sl_pts = pos["sl_points"]

        if pos["direction"] == 0:  # LONG
            # High is best case, low is worst case
            hit_tp = (bar_high - entry) >= tp_pts
            hit_sl = (bar_low - entry) <= -sl_pts
        else:  # SHORT
            # Low is best case, high is worst case
            hit_tp = (entry - bar_low) >= tp_pts
            hit_sl = (entry - bar_high) <= -sl_pts

        # If both hit in same bar, assume SL hit first (conservative)
        if hit_tp and hit_sl:
            hit_tp = False

        # Check barriers or timeout
        if hit_tp or hit_sl or pos["bars_held"] >= self.config.max_bars:
            # Calculate P&L based on exit price
            if hit_tp:
                exit_price = entry + tp_pts if pos["direction"] == 0 else entry - tp_pts
            elif hit_sl:
                exit_price = entry - sl_pts if pos["direction"] == 0 else entry + sl_pts
            else:  # TIMEOUT — exits at bar close
                exit_price = bar_close

            if pos["direction"] == 0:
                price_diff = exit_price - entry
            else:
                price_diff = entry - exit_price

            # P&L in dollars (NQ: $20/point)
            pnl = price_diff * pos["size"] * 20.0
            self.risk_mgr.record_trade_result(pnl, entry_bar=self._signal_count, confidence=pos.get("confidence", 0.0))

            result = "TP" if hit_tp else ("SL" if hit_sl else "TIMEOUT")
            ts_str = ts.strftime("%H:%M:%S")
            print(
                f"  [{ts_str}]   CLOSE {result} | "
                f"pnl=${pnl:+.2f} held={pos['bars_held']}bars | "
                f"equity={self.risk_mgr.state.current_equity:.2f}"
            )

            # Record result back into trade log and update CSVs
            idx = pos.get("trade_log_idx")
            if idx is not None and idx < len(self._trade_log):
                self._trade_log[idx]["result"] = result
                self._trade_log[idx]["pnl"] = pnl
                self._update_trade_row(idx)
                self._save_buckets()

            self._current_position = None

    def _save_trade_row(self, entry: dict) -> None:
        """Append a single trade row to the trades CSV."""
        row = pd.DataFrame([entry])
        row.to_csv(
            self._trades_csv,
            mode="a",
            header=not self._trades_csv_header_written,
            index=False,
        )
        self._trades_csv_header_written = True

    def _update_trade_row(self, idx: int) -> None:
        """Rewrite the full trades CSV after a result/pnl update."""
        pd.DataFrame(self._trade_log).to_csv(self._trades_csv, index=False)

    def _save_buckets(self) -> None:
        """Rewrite the buckets CSV with current stats."""
        df = pd.DataFrame(self._trade_log)
        trades = df[df["direction"] != "FLAT"].copy()
        if len(trades) == 0:
            return

        rows = []
        for lo in range(0, 100, 10):
            hi = lo + 10
            mask = (trades["confidence"] >= lo / 100) & (trades["confidence"] < hi / 100)
            bucket = trades[mask]
            if len(bucket) == 0:
                continue
            closed = bucket[bucket["result"].notna()] if "result" in bucket.columns else bucket.iloc[0:0]
            n_closed = len(closed)
            wins = int((closed["result"] == "TP").sum()) if n_closed > 0 else 0
            losses = int((closed["result"] == "SL").sum()) if n_closed > 0 else 0
            timeouts = int((closed["result"] == "TIMEOUT").sum()) if n_closed > 0 else 0
            bucket_pnl = closed["pnl"].sum() if n_closed > 0 and "pnl" in closed.columns else 0.0
            win_rate = (wins / n_closed * 100) if n_closed > 0 else 0.0
            rows.append({
                "bucket": f"{lo}-{hi}%",
                "trades": len(bucket),
                "closed": n_closed,
                "wins": wins,
                "losses": losses,
                "timeouts": timeouts,
                "win_rate": round(win_rate, 1),
                "pnl": round(bucket_pnl, 2),
            })

        if rows:
            # Add totals row
            t = pd.DataFrame(rows)
            rows.append({
                "bucket": "TOTAL",
                "trades": int(t["trades"].sum()),
                "closed": int(t["closed"].sum()),
                "wins": int(t["wins"].sum()),
                "losses": int(t["losses"].sum()),
                "timeouts": int(t["timeouts"].sum()),
                "win_rate": round(t["wins"].sum() / t["closed"].sum() * 100, 1) if t["closed"].sum() > 0 else 0.0,
                "pnl": round(t["pnl"].sum(), 2),
            })
            pd.DataFrame(rows).to_csv(self._buckets_csv, index=False)

    def run(self) -> None:
        """Start the live data stream and run inference loop."""
        print(f"\n{'='*60}")
        print(f"LIVE {'PAPER ' if self.paper else ''}TRADING")
        print(f"{'='*60}")
        print(f"  Dataset:  {self.dataset}")
        print(f"  Schema:   {self.schema}")
        print(f"  Symbols:  {', '.join(self.symbols)}")
        print(f"  Stype:    {self.stype_in}")
        print(f"  Lookback: {self.config.lookback} bars")
        print(f"  Buffer needs: {self.buffer.min_bars} bars before first signal")
        print(f"  Trades CSV:   {self._trades_csv}")
        print(f"  Buckets CSV:  {self._buckets_csv}")
        print(f"{'='*60}\n")

        max_retries = 10
        base_delay = 2  # seconds
        retries = 0

        try:
            while retries <= max_retries:
                try:
                    live_client = db.Live(key=self.api_key)
                    live_client.subscribe(
                        dataset=self.dataset,
                        schema=self.schema,
                        stype_in=self.stype_in,
                        symbols=self.symbols,
                    )

                    if retries == 0:
                        print("  Connecting to Databento live feed...")
                    else:
                        print(f"  Reconnecting (attempt {retries}/{max_retries})...")

                    print("  Connected! Waiting for bars...\n")
                    retries = 0  # reset on successful connection

                    for msg in live_client:
                        if hasattr(msg, "open") and hasattr(msg, "high") and hasattr(msg, "close"):
                            o = msg.open * self.PRICE_SCALE
                            h = msg.high * self.PRICE_SCALE
                            l = msg.low * self.PRICE_SCALE
                            c = msg.close * self.PRICE_SCALE
                            v = msg.volume

                            ts = pd.Timestamp(msg.ts_event, unit="ns", tz="UTC")
                            self._process_bar(ts.to_pydatetime(), o, h, l, c, v)
                            retries = 0  # reset on successful data

                        elif hasattr(msg, "err"):
                            print(f"\n  DATABENTO ERROR: {msg.err}")
                        elif hasattr(msg, "stype_in_symbol"):
                            print(f"  Symbol mapping: {msg.stype_in_symbol} -> instrument {msg.instrument_id}")
                        elif hasattr(msg, "msg"):
                            print(f"  DATABENTO SYSTEM: {msg.msg}")

                except db.common.error.BentoError as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"\n  Max retries ({max_retries}) exceeded. Giving up.")
                        raise
                    delay = min(base_delay * (2 ** (retries - 1)), 60)
                    print(f"\n  Connection lost: {e}")
                    print(f"  Reconnecting in {delay}s...")
                    time.sleep(delay)

        except KeyboardInterrupt:
            print("\n\n  Shutting down...")
        finally:
            self._print_summary()

    def _print_summary(self) -> None:
        """Print session summary."""
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Bars received:  {self._bar_count}")
        print(f"  Signals:        {self._signal_count}")
        print(f"  Trades logged:  {len(self._trade_log)}")

        risk = self.risk_mgr.summary()
        print(f"  Daily PnL:      ${risk['daily_pnl']:+.2f}")
        print(f"  Equity:         ${risk['current_equity']:.2f}")
        print(f"  Peak equity:    ${risk['peak_equity']:.2f}")
        print(f"  Drawdown:       ${risk['trailing_drawdown']:+.2f}")
        print(f"  Trades today:   {risk['trades_today']}")

        if self._trade_log:
            df = pd.DataFrame(self._trade_log)
            allowed = df[df["allowed"]]
            if len(allowed) > 0:
                print(f"\n  Signal breakdown (allowed trades):")
                print(f"    LONG:  {len(allowed[allowed['direction'] == 'LONG'])}")
                print(f"    SHORT: {len(allowed[allowed['direction'] == 'SHORT'])}")
                print(f"    Avg confidence: {allowed['confidence'].mean():.3f}")

            # Confidence bucket analysis
            trades = df[df["direction"] != "FLAT"].copy()
            if len(trades) > 0:
                print(f"\n  {'='*76}")
                print(f"  CONFIDENCE BUCKETS")
                print(f"  {'='*76}")
                print(f"  {'Bucket':<12} {'Trades':>6} {'Closed':>7} {'Wins':>5} {'Losses':>7} {'WinRate':>8} {'PnL':>10}")
                print(f"  {'-'*76}")

                total_closed = 0
                total_wins = 0
                total_losses = 0
                total_pnl = 0.0

                for lo in range(0, 100, 10):
                    hi = lo + 10
                    mask = (trades["confidence"] >= lo / 100) & (trades["confidence"] < hi / 100)
                    bucket = trades[mask]
                    if len(bucket) == 0:
                        continue

                    # Only count trades that have a result (closed positions)
                    closed = bucket[bucket["result"].notna()] if "result" in bucket.columns else bucket.iloc[0:0]
                    n_closed = len(closed)
                    wins = len(closed[closed["result"] == "TP"]) if n_closed > 0 else 0
                    losses = len(closed[closed["result"] == "SL"]) if n_closed > 0 else 0
                    bucket_pnl = closed["pnl"].sum() if n_closed > 0 and "pnl" in closed.columns else 0.0
                    win_rate = (wins / n_closed * 100) if n_closed > 0 else 0.0

                    total_closed += n_closed
                    total_wins += wins
                    total_losses += losses
                    total_pnl += bucket_pnl

                    print(
                        f"  {lo:>3}-{hi:<3}%    {len(bucket):>6} {n_closed:>7} {wins:>5} {losses:>7} "
                        f"{win_rate:>7.1f}% ${bucket_pnl:>+9.2f}"
                    )

                total_wr = (total_wins / total_closed * 100) if total_closed > 0 else 0.0
                print(f"  {'-'*76}")
                print(
                    f"  {'TOTAL':<12} {len(trades):>6} {total_closed:>7} {total_wins:>5} {total_losses:>7} "
                    f"{total_wr:>7.1f}% ${total_pnl:>+9.2f}"
                )

            # Final save (CSVs already updated incrementally)
            self._update_trade_row(0)  # full rewrite to ensure final state
            self._save_buckets()
            print(f"\n  Trade log:    {self._trades_csv}")
            print(f"  Bucket stats: {self._buckets_csv}")
