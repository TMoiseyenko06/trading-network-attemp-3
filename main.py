"""Main entry point — load data, configure GPU, train, and run inference."""

import warnings
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", message=".*torch.jit.script_method.*")
warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead to deadlocks.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

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
from neural_trading.live import LiveTrader
from neural_trading.backtest import Backtester
from neural_trading.optimizer import GridOptimizer


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV data from .dbn (Databento) or .csv files.

    For .dbn: uses databento library, expects ohlcv-* schema.
    For .csv: expects columns datetime index + open, high, low, close, volume.
    """
    path = Path(path)

    if path.suffix == ".dbn":
        import databento as db

        store = db.DBNStore.from_file(path)
        df = store.to_df()
        df.columns = [c.strip().lower() for c in df.columns]

        # Databento OHLCV prices are in fixed-point (int64 with 1e-9 scale)
        # to_df() already converts them to float, but verify we have the right cols
        required = {"open", "high", "low", "close", "volume"}
        available = set(df.columns)
        missing = required - available
        if missing:
            raise ValueError(
                f"Missing columns in .dbn file: {missing}. "
                f"Available: {sorted(available)}. "
                f"Make sure the file uses an ohlcv-* schema."
            )

        # Keep only OHLCV columns, drop extras like symbol, rtype, publisher_id
        df = df[["open", "high", "low", "close", "volume"]]

        # Index is already ts_event (DatetimeIndex) from to_df()
        df = df.sort_index()
        return df

    # Fallback: CSV
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
        max_bars=config.max_bars,
        lr=config.lr,
        epochs_per_fold=args.extra_epochs if args.resume else config.epochs_per_fold,
        patience=config.patience,
        tp_points=config.fixed_tp_points,
        sl_points=config.fixed_sl_points,
    )

    if args.resume:
        result = trainer.resume_training(
            df,
            checkpoint_path=args.resume,
            extra_epochs=args.extra_epochs,
        )
    else:
        result = trainer.train_backtest(df)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Backtest accuracy: {result['test_accuracy']:.3f}")
    print(f"  Backtest sortino:  {result['test_sortino']:.3f}")


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
    # Strip _orig_mod. prefix added by torch.compile()
    state = checkpoint["model_state"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    risk_mgr = RiskManager(
        RiskConfig(
            daily_loss_limit=config.daily_loss_limit,
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
        dir_logits, confidence, magnitude, pred_tp, pred_sl = model(X)

    probs = torch.softmax(dir_logits, dim=1).cpu().numpy()[0]
    conf = torch.sigmoid(confidence).cpu().item()
    mag = magnitude.cpu().item()
    tp = pred_tp.cpu().item()
    sl = pred_sl.cpu().item()
    direction = int(dir_logits.argmax(dim=1).cpu().item())

    last_close = df["close"].iloc[-1]
    labels = ["LONG (winner)", "SHORT (loser)", "FLAT (timeout)"]
    print(f"\nSignal: {labels[direction]}")
    print(f"  Probabilities: win={probs[0]:.3f} loss={probs[1]:.3f} flat={probs[2]:.3f}")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Expected magnitude: {mag:.5f}")
    print(f"  Predicted TP: {tp:.4%} (${last_close * tp:.2f})")
    print(f"  Predicted SL: {sl:.4%} (${last_close * sl:.2f})")

    allowed, size, reason = risk_mgr.check_trade(conf, direction, tp, sl)
    print(f"\nRisk check: {'APPROVED' if allowed else 'BLOCKED'}")
    print(f"  Position size: {size}")
    print(f"  Reason: {reason}")
    print(f"  Risk state: {risk_mgr.summary()}")


def backtest(args: argparse.Namespace) -> None:
    """Run realistic backtest with P&L simulation and equity curve."""
    config = Config()

    risk_cfg = RiskConfig(
        daily_loss_limit=config.daily_loss_limit,
        min_confidence=config.min_confidence,
        cooldown_bars=config.cooldown_bars,
        consecutive_loss_trigger=config.consecutive_loss_trigger,
        max_position_size=config.max_position_size,
        max_trades_per_day=config.max_trades_per_day,
        min_bars_between_trades=config.min_bars_between_trades,
    )
    if args.min_rr is not None:
        risk_cfg.min_rr_ratio = args.min_rr
    if args.min_confidence is not None:
        risk_cfg.min_confidence = args.min_confidence
    if args.no_halt:
        risk_cfg.no_halt = True

    bt = Backtester(
        model_path=args.model,
        point_value=args.point_value,
        starting_equity=args.equity,
        commission_per_contract=args.commission,
        max_bars_in_trade=config.max_bars,
        risk_config=risk_cfg,
        lookback=config.lookback,
        fixed_tp_points=config.fixed_tp_points,
        fixed_sl_points=config.fixed_sl_points,
    )

    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} bars from {args.data}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")

    result = bt.run(df, test_pct=args.test_pct)

    bt.plot_equity_curve(result, save_path=args.plot)
    bt.export_trades(result, save_path=args.trades)


def compare(args: argparse.Namespace) -> None:
    """Run multiple TP/SL setups on the same model and compare results."""
    config = Config()
    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} bars from {args.data}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")

    # Parse setups: "SL:TP" pairs
    setups = []
    for s in args.setups:
        sl_str, tp_str = s.split(":")
        setups.append((float(sl_str), float(tp_str)))

    min_conf = args.min_confidence

    results = {}
    for sl, tp in setups:
        label = f"SL{sl:.0f}_TP{tp:.0f}"
        print(f"\n{'#'*60}")
        print(f"  SETUP: {label}  (min conf={min_conf:.0%})")
        print(f"{'#'*60}")

        risk_cfg = RiskConfig(
            daily_loss_limit=config.daily_loss_limit,
            min_confidence=min_conf,
            cooldown_bars=config.cooldown_bars,
            consecutive_loss_trigger=config.consecutive_loss_trigger,
            max_position_size=config.max_position_size,
            max_trades_per_day=config.max_trades_per_day,
            min_bars_between_trades=config.min_bars_between_trades,
        )
        if args.no_halt:
            risk_cfg.no_halt = True

        bt = Backtester(
            model_path=args.model,
            point_value=args.point_value,
            starting_equity=args.equity,
            commission_per_contract=args.commission,
            max_bars_in_trade=config.max_bars,
            risk_config=risk_cfg,
            lookback=config.lookback,
            fixed_tp_points=tp,
            fixed_sl_points=sl,
        )

        result = bt.run(df, test_pct=args.test_pct)
        bt.plot_equity_curve(result, save_path=f"equity_{label}.png")
        bt.export_trades(result, save_path=f"trades_{label}.csv")
        results[label] = result

    # Print comparison table
    print(f"\n\n{'='*100}")
    print(f"  COMPARISON TABLE  (min confidence = {min_conf:.0%})")
    print(f"{'='*100}")

    header = (
        f"{'Setup':<14s} {'Trades':>6s} {'WR':>6s} {'PF':>5s} "
        f"{'Total P&L':>11s} {'Avg Win':>9s} {'Avg Loss':>9s} {'R:R':>5s} "
        f"{'MaxDD':>9s} {'DD%':>7s} {'Sharpe':>7s} {'Sortino':>7s} "
        f"{'ProfDays':>8s}"
    )
    print(header)
    print("-" * len(header))

    for label, r in results.items():
        row = (
            f"{label:<14s} {r.total_trades:>6d} {r.win_rate:>5.1%} {r.profit_factor:>5.2f} "
            f"${r.total_pnl:>+10,.0f} ${r.avg_win:>8,.0f} ${r.avg_loss:>8,.0f} {r.avg_rr_realized:>5.2f} "
            f"${r.max_drawdown:>8,.0f} {r.max_drawdown_pct:>6.1f}% {r.sharpe:>7.2f} {r.sortino:>7.2f} "
            f"{r.profitable_day_pct:>7.0f}%"
        )
        print(row)

    print(f"{'='*100}")

    # Rank by total P&L
    ranked = sorted(results.items(), key=lambda x: x[1].total_pnl, reverse=True)
    print(f"\n  Ranked by Total P&L:")
    for i, (label, r) in enumerate(ranked, 1):
        print(f"    {i}. {label:14s}  ${r.total_pnl:>+10,.0f}  (WR={r.win_rate:.1%}, PF={r.profit_factor:.2f}, Sharpe={r.sharpe:.2f})")

    # Rank by Sharpe
    ranked_sharpe = sorted(results.items(), key=lambda x: x[1].sharpe, reverse=True)
    print(f"\n  Ranked by Sharpe:")
    for i, (label, r) in enumerate(ranked_sharpe, 1):
        print(f"    {i}. {label:14s}  Sharpe={r.sharpe:.2f}  (P&L=${r.total_pnl:>+,.0f}, DD={r.max_drawdown_pct:.1f}%)")

    print()


def optimize(args: argparse.Namespace) -> None:
    """Run grid search over SL/TP/confidence combos."""
    config = Config()
    df = load_ohlcv(args.data)
    print(f"Loaded {len(df)} bars from {args.data}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")

    sl_range = [float(x) for x in np.arange(args.sl_min, args.sl_max + 1, args.step)]
    tp_range = [float(x) for x in np.arange(args.tp_min, args.tp_max + 1, args.step)]
    conf_range = [float(x) / 100 for x in range(args.conf_min, args.conf_max + 1, args.conf_step)]

    opt = GridOptimizer(
        model_path=args.model,
        point_value=args.point_value,
        starting_equity=args.equity,
        commission=args.commission,
        max_bars_in_trade=config.max_bars,
        lookback=config.lookback,
        test_pct=args.test_pct,
    )

    results_df = opt.run(df, sl_range, tp_range, conf_range)
    opt.print_top(results_df, n=args.top)

    # Save full results
    out_path = args.output
    results_df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to {out_path} ({len(results_df):,} combos)")


def live(args: argparse.Namespace) -> None:
    """Run live paper trading with Databento feed."""
    trader = LiveTrader(
        model_path=args.model,
        dataset=args.dataset,
        schema=args.schema,
        symbols=args.symbols,
        stype_in=args.stype,
        api_key=args.api_key,
    )
    trader.run()


def main():
    parser = argparse.ArgumentParser(description="Neural OHLCV Trading System")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Walk-forward train the model")
    train_p.add_argument("--data", required=True, help="Path to OHLCV data (.dbn or .csv)")
    train_p.add_argument("--resume", default=None, help="Path to checkpoint to resume from (e.g. model.pt)")
    train_p.add_argument("--extra-epochs", type=int, default=30, help="Additional epochs when resuming (default: 30)")

    infer_p = sub.add_parser("infer", help="Run inference on new data")
    infer_p.add_argument("--data", required=True, help="Path to OHLCV data (.dbn or .csv)")
    infer_p.add_argument("--model", required=True, help="Path to saved model checkpoint")

    bt_p = sub.add_parser("backtest", help="Run realistic backtest with P&L and equity curve")
    bt_p.add_argument("--data", required=True, help="Path to OHLCV data (.dbn or .csv)")
    bt_p.add_argument("--model", default="model.pt", help="Path to saved model checkpoint")
    bt_p.add_argument("--equity", type=float, default=50000.0, help="Starting equity (default: 50000)")
    bt_p.add_argument("--point-value", type=float, default=20.0, help="Point value per contract (NQ=20, ES=50)")
    bt_p.add_argument("--commission", type=float, default=4.50, help="Round-trip commission per contract")
    bt_p.add_argument("--test-pct", type=float, default=0.2, help="Fraction of data for test (default: 0.2)")
    bt_p.add_argument("--plot", default="equity_curve.png", help="Path to save equity curve plot")
    bt_p.add_argument("--trades", default="trades.csv", help="Path to save trade log CSV")
    bt_p.add_argument("--min-rr", type=float, default=None, help="Minimum R:R ratio (default: from RiskConfig)")
    bt_p.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence threshold (default: from RiskConfig)")
    bt_p.add_argument("--no-halt", action="store_true", help="Disable circuit breakers (daily loss limit & trailing drawdown halt)")

    cmp_p = sub.add_parser("compare", help="Run multiple TP/SL setups and compare results")
    cmp_p.add_argument("--data", required=True, help="Path to OHLCV data (.dbn or .csv)")
    cmp_p.add_argument("--model", default="model.pt", help="Path to saved model checkpoint")
    cmp_p.add_argument("--setups", nargs="+", required=True,
                        help="SL:TP pairs in points, e.g. 20:35 25:50 30:60")
    cmp_p.add_argument("--min-confidence", type=float, default=0.60,
                        help="Minimum confidence threshold (default: 0.60)")
    cmp_p.add_argument("--equity", type=float, default=50000.0, help="Starting equity")
    cmp_p.add_argument("--point-value", type=float, default=20.0, help="Point value (NQ=20)")
    cmp_p.add_argument("--commission", type=float, default=4.50, help="Round-trip commission")
    cmp_p.add_argument("--test-pct", type=float, default=0.2, help="Fraction of data for test")
    cmp_p.add_argument("--no-halt", action="store_true", help="Disable circuit breakers")

    opt_p = sub.add_parser("optimize", help="Grid search over SL/TP/confidence to find best params")
    opt_p.add_argument("--data", required=True, help="Path to OHLCV data (.dbn or .csv)")
    opt_p.add_argument("--model", default="model.pt", help="Path to saved model checkpoint")
    opt_p.add_argument("--sl-min", type=float, default=10, help="Min SL in points (default: 10)")
    opt_p.add_argument("--sl-max", type=float, default=100, help="Max SL in points (default: 100)")
    opt_p.add_argument("--tp-min", type=float, default=10, help="Min TP in points (default: 10)")
    opt_p.add_argument("--tp-max", type=float, default=100, help="Max TP in points (default: 100)")
    opt_p.add_argument("--step", type=float, default=5, help="Step size for SL/TP (default: 5)")
    opt_p.add_argument("--conf-min", type=int, default=0, help="Min confidence %% (default: 0)")
    opt_p.add_argument("--conf-max", type=int, default=90, help="Max confidence %% (default: 90)")
    opt_p.add_argument("--conf-step", type=int, default=5, help="Confidence step %% (default: 5)")
    opt_p.add_argument("--equity", type=float, default=50000.0, help="Starting equity")
    opt_p.add_argument("--point-value", type=float, default=20.0, help="Point value (NQ=20)")
    opt_p.add_argument("--commission", type=float, default=4.50, help="Round-trip commission")
    opt_p.add_argument("--test-pct", type=float, default=0.2, help="Fraction of data for test")
    opt_p.add_argument("--top", type=int, default=25, help="Number of top results to display")
    opt_p.add_argument("--output", default="grid_results.csv", help="Path to save full results CSV")

    live_p = sub.add_parser("live", help="Run live paper trading with Databento feed")
    live_p.add_argument("--model", default="model.pt", help="Path to saved model checkpoint")
    live_p.add_argument("--dataset", default="GLBX.MDP3", help="Databento dataset (default: GLBX.MDP3)")
    live_p.add_argument("--schema", default="ohlcv-1m", help="OHLCV schema (ohlcv-1s, ohlcv-1m, ohlcv-1h)")
    live_p.add_argument("--symbols", nargs="+", default=["NQ.c.0"], help="Symbols to subscribe to")
    live_p.add_argument("--stype", default="continuous", help="Symbol type (continuous, raw_symbol, etc.)")
    live_p.add_argument("--api-key", default=None, help="Databento API key (or set DATABENTO_API_KEY env)")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "backtest":
        backtest(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "optimize":
        optimize(args)
    elif args.command == "live":
        live(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
