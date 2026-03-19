"""Configuration dataclass for the full system."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All tunable parameters in one place."""

    # Data
    lookback: int = 90                  # bars of history the network sees
    max_bars: int = 10                  # forward window for labeling — shorter = faster resolution
    vol_window: int = 20                # rolling window for volume z-score

    # Training
    lr: float = 1e-3
    epochs_per_fold: int = 30
    patience: int = 7
    train_months: int = 3
    val_months: int = 1
    test_months: int = 1

    # Risk (selective high-conviction trading)
    daily_loss_limit: float = -500.0
    trailing_drawdown_limit: float = -2000.0
    min_confidence: float = 0.65        # only trade on high-conviction signals
    cooldown_bars: int = 5              # longer cooldown between trades
    consecutive_loss_trigger: int = 3   # tighter loss discipline
    max_position_size: int = 2
