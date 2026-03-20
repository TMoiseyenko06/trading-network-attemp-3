"""Configuration dataclass for the full system."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All tunable parameters in one place."""

    # Data
    lookback: int = 90                  # bars of history the network sees
    max_bars: int = 20                  # forward window for labeling — 20 min gives room for real moves
    vol_window: int = 20                # rolling window for volume z-score

    # Training
    lr: float = 1e-3
    epochs_per_fold: int = 100
    patience: int = 7
    train_months: int = 3
    val_months: int = 1
    test_months: int = 1

    # Risk (selective high-conviction trading — target 2-10 trades/day)
    daily_loss_limit: float = -500.0
    trailing_drawdown_limit: float = -2000.0
    min_confidence: float = 0.70        # high-conviction only
    cooldown_bars: int = 5              # longer cooldown between trades
    consecutive_loss_trigger: int = 3   # tighter loss discipline
    max_position_size: int = 1          # 1 contract until model proves profitable
    max_trades_per_day: int = 10        # hard cap on daily trade count
    min_bars_between_trades: int = 30   # ~30 min gap between entries on 1-min bars
