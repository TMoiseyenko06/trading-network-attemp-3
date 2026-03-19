"""
Custom loss function that combines:
  1. Cross-entropy on direction labels  (classification accuracy)
  2. Confidence calibration loss        (high confidence ↔ correct prediction)
  3. Magnitude regression loss          (Huber on expected move for directional trades)
  4. Drawdown penalty                   (penalise sequences of losses)
  5. Sortino-inspired penalty           (penalise downside deviation of simulated P&L)

Standard CE alone optimises for accuracy without caring about the *sequence*
of wins and losses — which is what kills you at prop firms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """
    All sub-losses are computed on the batch and returned individually for
    logging.  Weights are tunable.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        confidence_weight: float = 0.3,
        magnitude_weight: float = 0.2,
        drawdown_weight: float = 0.5,
        sortino_weight: float = 0.3,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.w_ce = ce_weight
        self.w_conf = confidence_weight
        self.w_mag = magnitude_weight
        self.w_dd = drawdown_weight
        self.w_sort = sortino_weight
        self.ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        confidence: torch.Tensor,          # (B, 1)
        magnitude: torch.Tensor,           # (B, 1)
        direction_labels: torch.Tensor,    # (B,) int64
        magnitude_labels: torch.Tensor,    # (B,) float
        close_returns: torch.Tensor,       # (B,) float — next-bar return for P&L sim
    ) -> tuple[torch.Tensor, dict[str, float]]:

        confidence = confidence.squeeze(-1)
        magnitude = magnitude.squeeze(-1)

        # 1. Classification
        ce_loss = self.ce_fn(direction_logits, direction_labels)

        # 2. Confidence calibration: confidence should be high when correct
        pred_dir = direction_logits.detach().argmax(dim=1)
        correct = (pred_dir == direction_labels).float()
        conf_loss = F.binary_cross_entropy(confidence, correct)

        # 3. Magnitude (only for directional trades — SL or TP, not timeout)
        dir_mask = direction_labels != 2
        if dir_mask.sum() > 0:
            mag_loss = F.huber_loss(magnitude[dir_mask], magnitude_labels[dir_mask])
        else:
            mag_loss = torch.tensor(0.0, device=direction_logits.device)

        # 4-5. Drawdown & Sortino on simulated P&L
        dd_loss, sort_loss = self._pnl_penalties(
            direction_logits, confidence, close_returns,
        )

        total = (
            self.w_ce * ce_loss
            + self.w_conf * conf_loss
            + self.w_mag * mag_loss
            + self.w_dd * dd_loss
            + self.w_sort * sort_loss
        )

        metrics = {
            "ce": ce_loss.item(),
            "conf": conf_loss.item(),
            "mag": mag_loss.item() if isinstance(mag_loss, torch.Tensor) else mag_loss,
            "dd": dd_loss.item(),
            "sortino": sort_loss.item(),
            "total": total.item(),
        }
        return total, metrics

    @staticmethod
    def _pnl_penalties(
        logits: torch.Tensor,
        confidence: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate per-bar P&L from predicted positions and penalise:
          - drawdown (peak-to-trough of cumulative P&L)
          - downside standard deviation (Sortino denominator)
        """
        probs = F.softmax(logits, dim=1)
        # position: long component minus short component, scaled by confidence
        # probs[:, 0]=SL  probs[:, 1]=TP  probs[:, 2]=Timeout
        # Treat TP as "long signal", SL as "short signal"
        position = (probs[:, 1] - probs[:, 0]) * confidence

        pnl = position * returns
        cum_pnl = torch.cumsum(pnl, dim=0)
        running_max = torch.cummax(cum_pnl, dim=0).values
        drawdown = running_max - cum_pnl
        dd_loss = drawdown.mean()

        # Sortino: downside deviation
        neg_pnl = torch.clamp(pnl, max=0.0)
        sort_loss = neg_pnl.std() + 1e-8

        return dd_loss, sort_loss
