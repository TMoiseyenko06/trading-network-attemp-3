#!/usr/bin/env python3
"""
NQ Scalping Neural Network — main entry point.

Usage
-----
  # Train on historical OHLCV CSV
  python main.py train --data nq_1min.csv

  # Train with custom barriers and lookback
  python main.py train --data nq_1min.csv --tp 35 --sl 20 --lookback 90

  # Run inference on a saved checkpoint
  python main.py infer --checkpoint checkpoints/fold_0_best.pt --data nq_live.csv

  # Force CPU (debugging)
  python main.py train --data nq_1min.csv --cpu

Data format
-----------
CSV with columns: timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from config import get_gpu_profile
from preprocessing import preprocess, NUM_FEATURES
from labels import compute_labels, label_stats
from training import Trainer, TrainConfig
from inference import LivePredictor
from risk_layer import RiskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def cmd_train(args: argparse.Namespace) -> None:
    """Full walk-forward training pipeline."""
    # GPU profile
    profile = get_gpu_profile(force_cpu=args.cpu)

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"  {len(df):,} bars loaded")

    # Labels (on raw prices, before preprocessing drops rows)
    logger.info(f"Computing triple-barrier labels (TP={args.tp}, SL={args.sl}, hold={args.max_hold})")
    dir_labels, mag_labels = compute_labels(
        df, tp_points=args.tp, sl_points=args.sl, max_holding=args.max_hold,
    )
    stats = label_stats(dir_labels)
    logger.info(f"  Label distribution: SL={stats['sl_pct']:.1f}%  TP={stats['tp_pct']:.1f}%  Timeout={stats['timeout_pct']:.1f}%")

    # Preprocess features
    logger.info("Preprocessing OHLCV → stationary features")
    features_df = preprocess(df, vol_lookback=args.vol_lookback)

    # Align labels to the feature index (preprocess drops some leading NaN rows)
    # features_df index maps back to original df rows
    valid_idx = features_df.index
    if isinstance(valid_idx, pd.DatetimeIndex):
        # Re-index: find which rows of original df survived
        df_indexed = df.copy()
        df_indexed.columns = [c.lower().strip() for c in df_indexed.columns]
        if "timestamp" in df_indexed.columns:
            df_indexed["timestamp"] = pd.to_datetime(df_indexed["timestamp"])
            df_indexed = df_indexed.set_index("timestamp")
        df_indexed = df_indexed.sort_index()
        mask = df_indexed.index.isin(valid_idx)
        dir_labels = dir_labels[mask]
        mag_labels = mag_labels[mask]
        close_rets = df_indexed["close"].pct_change().values[mask]
    else:
        # Numeric index fallback — preprocess drops the first few rows
        n_dropped = len(df) - len(features_df)
        dir_labels = dir_labels[n_dropped:]
        mag_labels = mag_labels[n_dropped:]
        close_rets = df["close"].pct_change().values[n_dropped:]

    features = features_df.values.astype(np.float32)
    dir_labels = dir_labels.astype(np.int64)
    mag_labels = mag_labels.astype(np.float32)
    close_rets = np.nan_to_num(close_rets, 0.0).astype(np.float32)

    logger.info(f"  {len(features):,} bars after preprocessing ({NUM_FEATURES} features)")

    # Train
    config = TrainConfig(
        lookback=args.lookback,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        tp_points=args.tp,
        sl_points=args.sl,
        max_holding=args.max_hold,
    )

    trainer = Trainer(
        features=features,
        direction_labels=dir_labels,
        magnitude_labels=mag_labels,
        close_returns=close_rets,
        profile=profile,
        config=config,
    )

    results = trainer.run()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD RESULTS")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"  Fold {r['fold']} | "
            f"val_loss={r['best_val_loss']:.4f}  "
            f"test_loss={r['test_loss']:.4f}  "
            f"test_acc={r['test_accuracy']:.2%}"
        )
    if results:
        avg_acc = np.mean([r["test_accuracy"] for r in results])
        logger.info(f"\n  Mean test accuracy: {avg_acc:.2%}")


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference bar-by-bar on a CSV (simulates live feed)."""
    profile = get_gpu_profile(force_cpu=args.cpu)

    risk_cfg = RiskConfig(
        daily_loss_limit=args.daily_loss_limit,
        trailing_dd_limit=args.trailing_dd_limit,
        min_confidence=args.min_confidence,
    )

    predictor = LivePredictor(
        checkpoint_path=args.checkpoint,
        profile=profile,
        risk_config=risk_cfg,
        lookback=args.lookback,
    )

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]
    logger.info(f"Simulating inference on {len(df):,} bars")

    decisions = []
    for _, row in df.iterrows():
        bar = row.to_dict()
        decision = predictor.on_bar(bar)
        decisions.append({
            "timestamp": bar.get("timestamp", ""),
            "signal": decision.signal.value,
            "size": decision.position_size,
            "confidence": decision.confidence,
            "magnitude": decision.magnitude,
            "blocked": decision.blocked,
            "reason": decision.block_reason,
        })

    out_df = pd.DataFrame(decisions)
    out_path = args.output or "inference_results.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    # Quick summary
    total = len(out_df)
    active = out_df[~out_df["blocked"]]
    longs = (active["signal"] == "long").sum()
    shorts = (active["signal"] == "short").sum()
    blocked = out_df["blocked"].sum()
    logger.info(f"  Total bars: {total}  Longs: {longs}  Shorts: {shorts}  Blocked: {blocked}")


def main():
    parser = argparse.ArgumentParser(
        description="NQ Scalping Neural Network — OHLCV only, GPU auto-scaled",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Walk-forward training")
    p_train.add_argument("--data", required=True, help="Path to OHLCV CSV")
    p_train.add_argument("--lookback", type=int, default=90)
    p_train.add_argument("--tp", type=float, default=35.0, help="Take-profit points")
    p_train.add_argument("--sl", type=float, default=20.0, help="Stop-loss points")
    p_train.add_argument("--max-hold", type=int, default=30, help="Max holding bars")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--train-months", type=int, default=4)
    p_train.add_argument("--val-months", type=int, default=1)
    p_train.add_argument("--test-months", type=int, default=1)
    p_train.add_argument("--vol-lookback", type=int, default=20)
    p_train.add_argument("--cpu", action="store_true", help="Force CPU")

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run inference on OHLCV CSV")
    p_infer.add_argument("--data", required=True, help="Path to OHLCV CSV")
    p_infer.add_argument("--checkpoint", required=True, help="Model checkpoint .pt")
    p_infer.add_argument("--lookback", type=int, default=90)
    p_infer.add_argument("--output", type=str, default=None, help="Output CSV path")
    p_infer.add_argument("--daily-loss-limit", type=float, default=1000.0)
    p_infer.add_argument("--trailing-dd-limit", type=float, default=2000.0)
    p_infer.add_argument("--min-confidence", type=float, default=0.55)
    p_infer.add_argument("--cpu", action="store_true", help="Force CPU")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)


if __name__ == "__main__":
    main()
