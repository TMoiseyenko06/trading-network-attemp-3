"""Main entry point — load data, configure GPU, train, and run inference."""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from neural_trading.gpu import detect_gpu
from neural_trading.config import Config
from neural_trading.model import NeuralOHLCVNet
from neural_trading.trainer import WalkForwardTrainer
from neural_trading.risk import RiskManager, RiskConfig
from neural_trading.preprocessing import compute_features, add_time_features, build_sequences


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV.

    Expected columns: datetime (or date), open, high, low, close, volume
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.sort_index()
    return df


def train(args: argparse.Namespace) -> None:
    """Train the model using walk-forward validation."""
    config = Config()
    gpu = detect_gpu()

    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} bars from {args.data}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")

    trainer = WalkForwardTrainer(
        gpu_profile=gpu,
        lookback=config.lookback,
        tp_pct=config.tp_pct,
        sl_pct=config.sl_pct,
        max_bars=config.max_bars,
        lr=config.lr,
        epochs_per_fold=config.epochs_per_fold,
        patience=config.patience,
    )

    results = trainer.walk_forward(
        df,
        train_months=config.train_months,
        val_months=config.val_months,
        test_months=config.test_months,
    )

    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)
    for r in results:
        print(f"  Fold {r['fold']}: acc={r['test_accuracy']:.3f} sortino={r['test_sortino']:.3f}")
    avg_acc = np.mean([r["test_accuracy"] for r in results])
    avg_sort = np.mean([r["test_sortino"] for r in results])
    print(f"  Average: acc={avg_acc:.3f} sortino={avg_sort:.3f}")


def infer(args: argparse.Namespace) -> None:
    """Run inference on new data using a trained model."""
    config = Config()
    gpu = detect_gpu()

    checkpoint = torch.load(args.model, map_location=gpu.device, weights_only=True)
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = NeuralOHLCVNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(gpu.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    risk_mgr = RiskManager(
        RiskConfig(
            daily_loss_limit=config.daily_loss_limit,
            trailing_drawdown_limit=config.trailing_drawdown_limit,
            min_confidence=config.min_confidence,
            cooldown_bars=config.cooldown_bars,
            consecutive_loss_trigger=config.consecutive_loss_trigger,
            max_position_size=config.max_position_size,
        )
    )

    df = load_ohlcv(args.data)
    features = compute_features(df)
    features = add_time_features(features)
    features = features.fillna(0)

    # Take the last lookback bars
    feat_vals = features.values[-config.lookback:].astype(np.float32)
    X = torch.from_numpy(feat_vals).unsqueeze(0).to(gpu.device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=gpu.use_amp):
        dir_logits, confidence, magnitude = model(X)

    probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
    conf = confidence.cpu().item()
    mag = magnitude.cpu().item()
    direction = int(dir_logits.argmax(dim=1).cpu().item())

    labels = ["LONG (winner)", "SHORT (loser)", "FLAT (timeout)"]
    print(f"\nSignal: {labels[direction]}")
    print(f"  Probabilities: win={probs[0]:.3f} loss={probs[1]:.3f} flat={probs[2]:.3f}")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Expected magnitude: {mag:.5f}")

    allowed, size, reason = risk_mgr.check_trade(conf, direction)
    print(f"\nRisk check: {'APPROVED' if allowed else 'BLOCKED'}")
    print(f"  Position size: {size}")
    print(f"  Reason: {reason}")
    print(f"  Risk state: {risk_mgr.summary()}")


def main():
    parser = argparse.ArgumentParser(description="Neural OHLCV Trading System")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Walk-forward train the model")
    train_p.add_argument("--data", required=True, help="Path to OHLCV CSV")

    infer_p = sub.add_parser("infer", help="Run inference on new data")
    infer_p.add_argument("--data", required=True, help="Path to OHLCV CSV")
    infer_p.add_argument("--model", required=True, help="Path to saved model checkpoint")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
