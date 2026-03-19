"""Custom loss combining classification, magnitude regression, drawdown penalty, and Sortino."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Multi-objective loss for the trading network.

    Components:
        1. Cross-entropy for direction classification
        2. MSE for magnitude prediction
        3. Sortino-inspired penalty: penalizes sequences of losses
        4. Confidence calibration: confidence should correlate with correctness
        5. TP/SL regression: Huber loss on predicted take-profit and stop-loss
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.3,
        sortino_weight: float = 0.5,
        confidence_weight: float = 0.2,
        tp_sl_weight: float = 0.5,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.sortino_weight = sortino_weight
        self.confidence_weight = confidence_weight
        self.tp_sl_weight = tp_sl_weight

    def forward(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        confidence: torch.Tensor,          # (B,)
        pred_magnitude: torch.Tensor,      # (B,)
        pred_tp: torch.Tensor,             # (B,)
        pred_sl: torch.Tensor,             # (B,)
        true_labels: torch.Tensor,         # (B,) int
        true_magnitude: torch.Tensor,      # (B,) float
        true_tp: torch.Tensor,             # (B,) float
        true_sl: torch.Tensor,             # (B,) float
    ) -> tuple[torch.Tensor, dict[str, float]]:

        # 1. Classification loss
        cls_loss = F.cross_entropy(direction_logits, true_labels)

        # 2. Magnitude regression loss
        mag_loss = F.mse_loss(pred_magnitude, true_magnitude)

        # 3. Sortino-inspired component
        preds = direction_logits.argmax(dim=1)
        correct = (preds == true_labels).float()
        pnl = torch.where(correct.bool(), true_magnitude, -true_magnitude)
        downside = torch.clamp(pnl, max=0)
        downside_std = downside.std() + 1e-8
        sortino = -(pnl.mean() / downside_std)  # negative because we minimize

        # 4. Confidence calibration
        conf_loss = F.binary_cross_entropy_with_logits(confidence, correct)

        # 5. TP/SL regression (Huber for robustness to outliers)
        tp_loss = F.huber_loss(pred_tp, true_tp, delta=0.01)
        sl_loss = F.huber_loss(pred_sl, true_sl, delta=0.01)
        tp_sl_loss = tp_loss + sl_loss

        total = (
            self.cls_weight * cls_loss
            + self.mag_weight * mag_loss
            + self.sortino_weight * sortino
            + self.confidence_weight * conf_loss
            + self.tp_sl_weight * tp_sl_loss
        )

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": mag_loss.item(),
            "sortino": -sortino.item(),
            "conf_loss": conf_loss.item(),
            "tp_sl_loss": tp_sl_loss.item(),
            "accuracy": correct.mean().item(),
        }
        return total, metrics
