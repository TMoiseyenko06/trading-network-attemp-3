"""Configuration dataclass for the full system."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """All tunable parameters in one place."""

    # Data
    lookback: int = 90                  # bars of history the network sees
    max_bars: int = 20                  # forward window for labeling (network predicts its own TP/SL)
    vol_window: int = 20                # rolling window for volume z-score

    # Training
    lr: float = 1e-3
    epochs_per_fold: int = 30
    patience: int = 7
    train_months: int = 3
    val_months: int = 1
    test_months: int = 1

    # Risk (prop firm survival)
    daily_loss_limit: float = -1250.0
    trailing_drawdown_limit: float = -2250.0
    min_confidence: float = 0.6
    cooldown_bars: int = 5
    consecutive_loss_trigger: int = 3
    max_position_size: int = 4
