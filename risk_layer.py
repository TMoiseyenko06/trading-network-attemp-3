"""
Risk management layer — hard-coded rules the network CANNOT override.

This is non-negotiable for prop firm survival.  It sits *outside* the model
and gates every trade signal before execution.

Rules
-----
1. Daily loss limit circuit breaker
2. Trailing drawdown tracker (scales position size down as you approach limits)
3. Minimum confidence threshold (filter low-conviction signals)
4. Cooldown period after consecutive losses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TradeDecision:
    """Output of the risk layer — what the execution engine should do."""
    signal: Signal
    position_size: float       # 0.0 – 1.0 (fraction of max size)
    confidence: float
    magnitude: float
    blocked: bool = False
    block_reason: str = ""


@dataclass
class RiskConfig:
    # Daily loss limit (in account currency / points)
    daily_loss_limit: float = 1000.0

    # Trailing drawdown: max peak-to-trough before full shutdown
    trailing_dd_limit: float = 2000.0

    # Position scaling: start reducing size when drawdown exceeds this % of limit
    dd_scale_start_pct: float = 0.5    # start scaling at 50% of limit
    dd_scale_min_size: float = 0.25    # minimum position size at limit edge

    # Confidence gating
    min_confidence: float = 0.55       # ignore signals below this

    # Cooldown
    max_consecutive_losses: int = 3    # pause after this many losses in a row
    cooldown_bars: int = 10            # bars to wait during cooldown


@dataclass
class RiskState:
    """Mutable state tracked across the trading session."""
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    trades_today: int = 0
    is_halted: bool = False
    halt_reason: str = ""


class RiskLayer:
    """
    Stateful risk manager.  Call `evaluate()` on every bar with the model's
    raw output.  Call `update()` after each trade result.
    """

    def __init__(self, config: RiskConfig | None = None, starting_equity: float = 0.0):
        self.cfg = config or RiskConfig()
        self.state = RiskState(
            peak_equity=starting_equity,
            current_equity=starting_equity,
        )

    def evaluate(
        self,
        direction_probs: list[float],   # [p_sl, p_tp, p_timeout]
        confidence: float,
        magnitude: float,
    ) -> TradeDecision:
        """
        Gate the model's output through all risk rules.

        Parameters
        ----------
        direction_probs : softmax output [P(SL), P(TP), P(Timeout)]
        confidence      : model's confidence score (0-1)
        magnitude       : expected price move

        Returns
        -------
        TradeDecision with signal, position_size, and blocking info
        """
        # Determine raw signal from probabilities
        signal = self._raw_signal(direction_probs)

        decision = TradeDecision(
            signal=signal,
            position_size=1.0,
            confidence=confidence,
            magnitude=magnitude,
        )

        # --- Rule 1: Daily loss limit circuit breaker ----------------------
        if self.state.is_halted:
            return self._block(decision, f"HALTED: {self.state.halt_reason}")

        if self.state.daily_pnl <= -self.cfg.daily_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = "daily loss limit hit"
            return self._block(decision, "Daily loss limit breached")

        # --- Rule 2: Trailing drawdown position scaling --------------------
        dd = self.state.peak_equity - self.state.current_equity
        dd_pct = dd / self.cfg.trailing_dd_limit if self.cfg.trailing_dd_limit > 0 else 0

        if dd_pct >= 1.0:
            self.state.is_halted = True
            self.state.halt_reason = "trailing drawdown limit hit"
            return self._block(decision, "Trailing drawdown limit breached")

        if dd_pct >= self.cfg.dd_scale_start_pct:
            # Linearly scale down from 1.0 to dd_scale_min_size
            scale_range = 1.0 - self.cfg.dd_scale_start_pct
            scale_progress = (dd_pct - self.cfg.dd_scale_start_pct) / scale_range
            decision.position_size = max(
                self.cfg.dd_scale_min_size,
                1.0 - scale_progress * (1.0 - self.cfg.dd_scale_min_size),
            )

        # --- Rule 3: Minimum confidence threshold -------------------------
        if confidence < self.cfg.min_confidence:
            return self._block(decision, f"Confidence {confidence:.2f} < {self.cfg.min_confidence}")

        # --- Rule 4: Cooldown after consecutive losses ---------------------
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._block(decision, f"Cooldown ({self.state.cooldown_remaining} bars remaining)")

        if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.state.cooldown_remaining = self.cfg.cooldown_bars
            self.state.consecutive_losses = 0
            return self._block(decision, "Cooldown triggered after consecutive losses")

        # --- Signal is flat → no trade, but not "blocked" ------------------
        if signal == Signal.FLAT:
            decision.position_size = 0.0

        return decision

    def update(self, pnl: float) -> None:
        """Call after every closed trade with the realized P&L."""
        self.state.daily_pnl += pnl
        self.state.current_equity += pnl
        self.state.trades_today += 1

        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        logger.debug(
            f"Trade result: PnL={pnl:+.2f}  DailyPnL={self.state.daily_pnl:+.2f}  "
            f"DD={self.state.peak_equity - self.state.current_equity:.2f}  "
            f"ConsecL={self.state.consecutive_losses}"
        )

    def reset_daily(self) -> None:
        """Call at the start of each trading day."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self.state.is_halted = False
        self.state.halt_reason = ""
        self.state.cooldown_remaining = 0

    @staticmethod
    def _raw_signal(probs: list[float]) -> Signal:
        """Convert model probabilities to a directional signal."""
        p_sl, p_tp, p_timeout = probs
        if p_tp > p_sl and p_tp > p_timeout:
            return Signal.LONG
        elif p_sl > p_tp and p_sl > p_timeout:
            return Signal.SHORT
        else:
            return Signal.FLAT

    @staticmethod
    def _block(decision: TradeDecision, reason: str) -> TradeDecision:
        decision.blocked = True
        decision.block_reason = reason
        decision.signal = Signal.FLAT
        decision.position_size = 0.0
        return decision
