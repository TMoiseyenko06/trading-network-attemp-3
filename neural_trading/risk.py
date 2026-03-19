"""Hard-coded risk management layer that sits outside the neural network."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class RiskConfig:
    """Risk parameters — these are non-negotiable hard limits."""
    daily_loss_limit: float = -500.0          # max daily loss in dollars
    trailing_drawdown_limit: float = -2000.0  # max trailing drawdown
    min_confidence: float = 0.6               # minimum network confidence to trade
    cooldown_bars: int = 5                    # bars to wait after consecutive losses
    consecutive_loss_trigger: int = 3         # losses before cooldown activates
    max_position_size: int = 4                # max contracts
    drawdown_scale_start: float = 0.5         # start scaling at 50% of drawdown limit
    min_rr_ratio: float = 0.8                 # minimum risk/reward ratio (TP/SL)


@dataclass
class RiskState:
    """Mutable state tracked by the risk manager."""
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    trailing_drawdown: float = 0.0
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    trades_today: int = 0
    is_halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """Enforces hard risk limits that the neural network cannot override."""

    def __init__(self, config: Optional[RiskConfig] = None, starting_equity: float = 50000.0):
        self.config = config or RiskConfig()
        self.state = RiskState(
            peak_equity=starting_equity,
            current_equity=starting_equity,
        )

    def check_trade(
        self, confidence: float, predicted_direction: int,
        tp_pct: float = 0.0, sl_pct: float = 0.0,
    ) -> tuple[bool, int, str]:
        """Decide whether a trade is allowed and compute position size.

        Args:
            confidence: network confidence score [0, 1]
            predicted_direction: 0=long winner, 1=short loser, 2=flat/timeout
            tp_pct: predicted take-profit as decimal percentage
            sl_pct: predicted stop-loss as decimal percentage

        Returns:
            (allowed, position_size, reason)
        """
        # Circuit breaker: daily loss limit
        if self.state.daily_pnl <= self.config.daily_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = "daily_loss_limit"
            return False, 0, "Daily loss limit reached"

        # Circuit breaker: trailing drawdown
        if self.state.trailing_drawdown <= self.config.trailing_drawdown_limit:
            self.state.is_halted = True
            self.state.halt_reason = "trailing_drawdown"
            return False, 0, "Trailing drawdown limit reached"

        if self.state.is_halted:
            return False, 0, f"Trading halted: {self.state.halt_reason}"

        # Don't trade flat/timeout signals
        if predicted_direction == 2:
            return False, 0, "Signal is flat/timeout — no trade"

        # Minimum confidence filter
        if confidence < self.config.min_confidence:
            return False, 0, f"Confidence {confidence:.2f} below minimum {self.config.min_confidence}"

        # Minimum risk/reward ratio — don't take trades risking more than potential gain
        if tp_pct > 0 and sl_pct > 0:
            rr_ratio = tp_pct / sl_pct
            if rr_ratio < self.config.min_rr_ratio:
                return False, 0, f"R:R {rr_ratio:.2f} below minimum {self.config.min_rr_ratio}"

        # Cooldown after consecutive losses
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return False, 0, f"Cooldown: {self.state.cooldown_remaining} bars remaining"

        # Position sizing — scale down as we approach drawdown limit
        dd_ratio = abs(self.state.trailing_drawdown / self.config.trailing_drawdown_limit)
        if dd_ratio >= self.config.drawdown_scale_start:
            # Linear scale from max to 1 contract
            scale = 1.0 - (dd_ratio - self.config.drawdown_scale_start) / (1.0 - self.config.drawdown_scale_start)
            size = max(1, int(self.config.max_position_size * scale * confidence))
        else:
            size = max(1, int(self.config.max_position_size * confidence))

        return True, size, "Trade approved"

    def record_trade_result(self, pnl: float) -> None:
        """Update state after a trade completes."""
        self.state.daily_pnl += pnl
        self.state.current_equity += pnl
        self.state.trades_today += 1

        # Update peak and trailing drawdown
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity
        self.state.trailing_drawdown = self.state.current_equity - self.state.peak_equity

        # Track consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.config.consecutive_loss_trigger:
                self.state.cooldown_remaining = self.config.cooldown_bars
        else:
            self.state.consecutive_losses = 0

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self.state.is_halted = False
        self.state.halt_reason = ""

    def summary(self) -> dict:
        """Return current risk state as a dict."""
        return {
            "daily_pnl": self.state.daily_pnl,
            "trailing_drawdown": self.state.trailing_drawdown,
            "peak_equity": self.state.peak_equity,
            "current_equity": self.state.current_equity,
            "consecutive_losses": self.state.consecutive_losses,
            "cooldown_remaining": self.state.cooldown_remaining,
            "is_halted": self.state.is_halted,
            "trades_today": self.state.trades_today,
        }
