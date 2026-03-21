"""Configuration dataclass for the full system."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All tunable parameters in one place."""

    # Data
    lookback: int = 90                  # bars of history the network sees
    max_bars: int = 20                  # forward window for labeling — 20 min gives room for real moves
    vol_window: int = 20                # rolling window for volume z-score

    # Fixed TP/SL (hard rules — not predicted by the network)
    fixed_tp_points: float = 35.0       # take-profit in index points
    fixed_sl_points: float = 20.0       # stop-loss in index points

    # Training
    lr: float = 1e-3
    epochs_per_fold: int = 100
    patience: int = 7
    train_months: int = 3
    val_months: int = 1
    test_months: int = 1

    # Risk (active trading — target 3-10 trades/day, most days)
    daily_loss_limit: float = -1500.0   # allow ~3 consecutive losses before halt
    trailing_drawdown_limit: float = -3000.0
    min_confidence: float = 0.60        # moderate conviction threshold
    cooldown_bars: int = 5              # cooldown after consecutive losses
    consecutive_loss_trigger: int = 3   # losses before cooldown activates
    max_position_size: int = 1          # 1 contract until model proves profitable
    max_trades_per_day: int = 10        # more opportunities per day
    min_bars_between_trades: int = 15   # ~15 min gap between entries on 1-min bars
