"""Custom loss combining classification, magnitude regression, and auxiliary objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Multi-objective loss for the trading network.

    Components:
        1. Cross-entropy for direction classification (with optional class weights)
        2. MSE for magnitude prediction
        3. PnL-aware Sortino using predicted TP/SL for realistic profit estimation
           (TP/SL detached — Sortino only teaches direction, not TP/SL values)
        4. Confidence calibration via soft correctness probabilities
        5. TP/SL regression: Huber loss on predicted take-profit and stop-loss
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.3,
        sortino_weight: float = 0.3,
        confidence_weight: float = 0.2,
        tp_sl_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.sortino_weight = sortino_weight
        self.confidence_weight = confidence_weight
        self.tp_sl_weight = tp_sl_weight
        self.register_buffer("class_weights", class_weights)

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

        # 1. Classification loss (with class weights to handle imbalance)
        cls_loss = F.cross_entropy(direction_logits, true_labels, weight=self.class_weights)

        # 2. Magnitude regression loss
        mag_loss = F.mse_loss(pred_magnitude, true_magnitude)

        # 3. PnL-aware Sortino using predicted TP/SL
        # Detach TP/SL so Sortino only trains direction, not TP/SL values.
        # Without detach, the model learns to max TP / min SL regardless of data
        # because that directly increases the Sortino numerator.
        probs = F.softmax(direction_logits.float(), dim=1)  # (B, 3)
        true_onehot = F.one_hot(true_labels, num_classes=direction_logits.shape[1]).float()
        soft_correct = (probs * true_onehot).sum(dim=1)  # (B,) in [0, 1]

        # Realistic PnL: win → earn TP, lose → pay SL
        tp_detached = pred_tp.detach().float()
        sl_detached = pred_sl.detach().float()
        soft_pnl = soft_correct * tp_detached - (1.0 - soft_correct) * sl_detached
        downside = torch.clamp(soft_pnl, max=0)
        downside_var = (downside ** 2).mean()
        # Larger epsilon prevents gradient explosion when all PnLs are positive
        # (downside_var ≈ 0 → sqrt gradient → 1/(2*sqrt(eps)) → huge under AMP)
        downside_std = torch.sqrt(downside_var + 1e-4)
        sortino = -(soft_pnl.mean() / downside_std).clamp(-10, 10)

        # 4. Confidence calibration (soft correctness as target)
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence, soft_correct.detach(),
        )

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

        # Accuracy metric (hard, for reporting only)
        hard_preds = direction_logits.argmax(dim=1)
        accuracy = (hard_preds == true_labels).float().mean().item()

        # R:R for monitoring only (not in the loss)
        rr_ratio = pred_tp / (pred_sl + 1e-8)

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": mag_loss.item(),
            "sortino": -sortino.item(),
            "conf_loss": conf_loss.item(),
            "tp_sl_loss": tp_sl_loss.item(),
            "rr_ratio": rr_ratio.mean().item(),
            "accuracy": accuracy,
        }
        return total, metrics
