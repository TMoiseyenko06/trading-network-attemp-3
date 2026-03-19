"""
Live inference pipeline.

Takes raw OHLCV bars, preprocesses, runs through the model, then passes
the output through the risk layer to produce a final trade decision.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import GPUProfile, get_gpu_profile
from preprocessing import preprocess, FEATURE_COLS, NUM_FEATURES
from model import build_model
from risk_layer import RiskLayer, RiskConfig, TradeDecision
import pandas as pd

logger = logging.getLogger(__name__)


class LivePredictor:
    """
    Maintains a rolling buffer of preprocessed bars and produces
    trade decisions on each new bar.
    """

    def __init__(
        self,
        checkpoint_path: str,
        profile: GPUProfile | None = None,
        risk_config: RiskConfig | None = None,
        lookback: int = 90,
        vol_lookback: int = 20,
        starting_equity: float = 50_000.0,
    ):
        self.profile = profile or get_gpu_profile()
        self.lookback = lookback
        self.vol_lookback = vol_lookback

        # Build model and load weights
        self.model = build_model(NUM_FEATURES, self.profile)
        state = torch.load(checkpoint_path, map_location=self.profile.device, weights_only=True)
        # Handle torch.compile wrapped models
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        # Risk layer
        self.risk = RiskLayer(risk_config, starting_equity=starting_equity)

        # Rolling bar buffer (raw OHLCV for preprocessing context)
        self._bar_buffer: deque[dict] = deque(maxlen=lookback + vol_lookback + 10)

    def on_bar(self, bar: dict) -> TradeDecision:
        """
        Process a new OHLCV bar and return a trade decision.

        Parameters
        ----------
        bar : dict with keys: timestamp, open, high, low, close, volume

        Returns
        -------
        TradeDecision
        """
        self._bar_buffer.append(bar)

        if len(self._bar_buffer) < self.lookback + self.vol_lookback:
            from risk_layer import Signal
            return TradeDecision(
                signal=Signal.FLAT,
                position_size=0.0,
                confidence=0.0,
                magnitude=0.0,
                blocked=True,
                block_reason=f"Warming up ({len(self._bar_buffer)}/{self.lookback + self.vol_lookback} bars)",
            )

        # Preprocess the buffer into features
        df = pd.DataFrame(list(self._bar_buffer))
        features_df = preprocess(df, vol_lookback=self.vol_lookback)

        if len(features_df) < self.lookback:
            from risk_layer import Signal
            return TradeDecision(
                signal=Signal.FLAT,
                position_size=0.0,
                confidence=0.0,
                magnitude=0.0,
                blocked=True,
                block_reason="Not enough valid bars after preprocessing",
            )

        # Take the last `lookback` bars
        window = features_df.values[-self.lookback:]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        x = x.to(self.profile.device)

        # Forward pass
        with torch.no_grad(), torch.autocast(
            device_type="cuda" if self.profile.device.type == "cuda" else "cpu",
            dtype=self.profile.amp_dtype,
            enabled=self.profile.use_amp,
        ):
            dir_logits, conf, mag = self.model(x)

        probs = F.softmax(dir_logits, dim=1).squeeze(0).cpu().tolist()
        confidence = conf.squeeze().cpu().item()
        magnitude = mag.squeeze().cpu().item()

        # Risk layer gating
        decision = self.risk.evaluate(probs, confidence, magnitude)

        logger.info(
            f"Bar {bar.get('timestamp', '?')} | "
            f"P(SL)={probs[0]:.2f} P(TP)={probs[1]:.2f} P(TO)={probs[2]:.2f} | "
            f"Conf={confidence:.2f} Mag={magnitude:.1f} | "
            f"Signal={decision.signal.value} Size={decision.position_size:.2f}"
            + (f" BLOCKED: {decision.block_reason}" if decision.blocked else "")
        )

        return decision

    def on_trade_closed(self, pnl: float) -> None:
        """Update the risk layer after a trade closes."""
        self.risk.update(pnl)

    def on_new_day(self) -> None:
        """Reset daily risk counters."""
        self.risk.reset_daily()
