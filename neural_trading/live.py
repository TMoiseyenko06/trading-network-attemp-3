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

        # Risk manager
        self.risk_mgr = RiskManager(
            RiskConfig(
                daily_loss_limit=self.config.daily_loss_limit,
                trailing_drawdown_limit=self.config.trailing_drawdown_limit,
                min_confidence=self.config.min_confidence,
                cooldown_bars=self.config.cooldown_bars,
                consecutive_loss_trigger=self.config.consecutive_loss_trigger,
                max_position_size=self.config.max_position_size,
            )
        )

        # Trade tracking
        self._trade_log: list[dict] = []
        self._current_position: Optional[dict] = None
        self._bar_count = 0
        self._signal_count = 0
        self._last_day: Optional[int] = None

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
            if self._bar_count % 10 == 0:
                print(f"  Buffering... {self.buffer.bar_count}/{self.buffer.min_bars} bars")
            return

        # Run inference
        signal = self._infer()
        if signal is not None:
            self._handle_signal(signal, ts)

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
        tp = pred_tp.cpu().item()
        sl = pred_sl.cpu().item()
        direction = int(dir_logits.argmax(dim=1).cpu().item())

        return {
            "direction": direction,
            "probs": probs,
            "confidence": conf,
            "magnitude": mag,
            "tp_pct": tp,
            "sl_pct": sl,
        }

    def _handle_signal(self, signal: dict, ts: datetime) -> None:
        """Process a model signal through risk management and log it."""
        direction = signal["direction"]
        conf = signal["confidence"]
        probs = signal["probs"]
        labels = ["LONG", "SHORT", "FLAT"]
        last_close = self.buffer.last_close

        # Check with risk manager
        allowed, size, reason = self.risk_mgr.check_trade(conf, direction)

        self._signal_count += 1

        # Resolve any open position (simulated)
        if self._current_position is not None:
            self._resolve_position(last_close, ts)

        # Log the signal
        ts_str = ts.strftime("%H:%M:%S")
        dir_label = labels[direction]
        prob_str = f"W={probs[0]:.2f} L={probs[1]:.2f} F={probs[2]:.2f}"

        if allowed:
            tp_price = last_close * (1 + signal["tp_pct"])
            sl_price = last_close * (1 - signal["sl_pct"])

            print(
                f"  [{ts_str}] #{self._signal_count:4d} "
                f"{dir_label:5s} x{size} @ {last_close:.2f} | "
                f"conf={conf:.2f} ({prob_str}) | "
                f"TP={tp_price:.2f} SL={sl_price:.2f} | "
                f"mag={signal['magnitude']:.4f}"
            )

            if self.paper:
                self._current_position = {
                    "direction": direction,
                    "entry_price": last_close,
                    "size": size,
                    "tp_pct": signal["tp_pct"],
                    "sl_pct": signal["sl_pct"],
                    "entry_time": ts,
                    "bars_held": 0,
                }
        else:
            # Only log blocked trades if they were actionable (not FLAT)
            if direction != 2:
                print(
                    f"  [{ts_str}] #{self._signal_count:4d} "
                    f"{dir_label:5s} BLOCKED | {reason} | "
                    f"conf={conf:.2f} ({prob_str})"
                )

        # Log trade for analysis
        self._trade_log.append({
            "timestamp": ts,
            "direction": dir_label,
            "confidence": conf,
            "probs_win": probs[0],
            "probs_lose": probs[1],
            "probs_flat": probs[2],
            "magnitude": signal["magnitude"],
            "tp_pct": signal["tp_pct"],
            "sl_pct": signal["sl_pct"],
            "price": last_close,
            "allowed": allowed,
            "size": size if allowed else 0,
            "reason": reason,
        })

    def _resolve_position(self, current_price: float, ts: datetime) -> None:
        """Simulate resolving the previous position (paper trading)."""
        pos = self._current_position
        if pos is None:
            return

        pos["bars_held"] += 1
        entry = pos["entry_price"]

        if pos["direction"] == 0:  # LONG
            ret = (current_price - entry) / entry
            hit_tp = ret >= pos["tp_pct"]
            hit_sl = ret <= -pos["sl_pct"]
        else:  # SHORT
            ret = (entry - current_price) / entry
            hit_tp = ret >= pos["tp_pct"]
            hit_sl = ret <= -pos["sl_pct"]

        # Check barriers or timeout
        if hit_tp or hit_sl or pos["bars_held"] >= self.config.max_bars:
            # Estimate PnL (simplified: per contract, point value varies by instrument)
            pnl = ret * pos["size"] * 1000  # rough NQ point value
            self.risk_mgr.record_trade_result(pnl)

            result = "TP" if hit_tp else ("SL" if hit_sl else "TIMEOUT")
            ts_str = ts.strftime("%H:%M:%S")
            print(
                f"  [{ts_str}]   CLOSE {result} | "
                f"pnl=${pnl:+.2f} held={pos['bars_held']}bars | "
                f"equity={self.risk_mgr.state.current_equity:.2f}"
            )
            self._current_position = None

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
        print(f"{'='*60}\n")

        live_client = db.Live(key=self.api_key)

        live_client.subscribe(
            dataset=self.dataset,
            schema=self.schema,
            stype_in=self.stype_in,
            symbols=self.symbols,
        )

        print("  Connecting to Databento live feed...")
        try:
            print("  Connected! Waiting for bars...\n")

            for msg in live_client:
                if hasattr(msg, "open") and hasattr(msg, "high") and hasattr(msg, "close"):
                    o = msg.open * self.PRICE_SCALE
                    h = msg.high * self.PRICE_SCALE
                    l = msg.low * self.PRICE_SCALE
                    c = msg.close * self.PRICE_SCALE
                    v = msg.volume

                    ts = pd.Timestamp(msg.ts_event, unit="ns", tz="UTC")
                    self._process_bar(ts.to_pydatetime(), o, h, l, c, v)

                elif hasattr(msg, "err"):
                    print(f"\n  DATABENTO ERROR: {msg.err}")
                elif hasattr(msg, "stype_in_symbol"):
                    print(f"  Symbol mapping: {msg.stype_in_symbol} -> instrument {msg.instrument_id}")
                elif hasattr(msg, "msg"):
                    print(f"  DATABENTO SYSTEM: {msg.msg}")

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

            # Save trade log
            log_path = f"live_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(log_path, index=False)
            print(f"\n  Trade log saved to {log_path}")
