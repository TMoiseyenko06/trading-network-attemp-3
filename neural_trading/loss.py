"""Custom loss for fixed TP/SL trading regime.

The model predicts direction (LONG/SHORT/FLAT) + confidence.
TP/SL are hard-coded (30pt TP, 20pt SL → 1.5 R:R).

Target: 75% trade win rate. Two loss components:
  1. Label-smoothed cross-entropy — calibrated direction prediction
  2. Wrong-trade penalty — extra cost when model predicts LONG/SHORT
     but the true label disagrees (teaches model to go FLAT when unsure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Classification loss targeting high win rate on trades.

    Cross-entropy with label smoothing + asymmetric penalty for wrong
    directional calls. The model should predict FLAT when uncertain
    rather than guessing wrong on a LONG/SHORT.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.0,
        sortino_weight: float = 0.0,
        confidence_weight: float = 0.0,
        pnl_weight: float = 0.0,
        frequency_weight: float = 3.0,
        tp_points: float = 30.0,
        sl_points: float = 20.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        wrong_trade_penalty: float = 0.5,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.frequency_weight = frequency_weight
        self.tp_points = tp_points
        self.sl_points = sl_points
        self.label_smoothing = label_smoothing
        self.wrong_trade_penalty = wrong_trade_penalty
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        confidence: torch.Tensor,          # (B,)
        pred_magnitude: torch.Tensor,      # (B,) — ignored
        pred_tp: torch.Tensor,             # (B,) — ignored
        pred_sl: torch.Tensor,             # (B,) — ignored
        true_labels: torch.Tensor,         # (B,) int
        true_magnitude: torch.Tensor,      # (B,) float — ignored
        true_tp: torch.Tensor,             # (B,) float — ignored
        true_sl: torch.Tensor,             # (B,) float — ignored
    ) -> tuple[torch.Tensor, dict[str, float]]:

        # Force float32, clamp to prevent overflow
        direction_logits = direction_logits.float().clamp(-50, 50)
        confidence = confidence.float().clamp(-50, 50)

        # 1. Label-smoothed cross-entropy — better calibration
        cls_loss = F.cross_entropy(
            direction_logits, true_labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # Probabilities for metrics and penalty
        probs = F.softmax(direction_logits, dim=1)  # (B, 3)
        hard_preds = direction_logits.argmax(dim=1)
        p_flat_mean = probs[:, 2].mean()

        # 2. Wrong-trade penalty — extra loss when model predicts LONG/SHORT
        #    but gets the direction wrong. This teaches the model: "if you're
        #    not sure, predict FLAT instead of guessing wrong."
        #    Only penalizes non-FLAT predictions that are incorrect.
        trade_mask = hard_preds != 2                          # predicted a direction
        wrong_mask = trade_mask & (hard_preds != true_labels) # but got it wrong

        if wrong_mask.any():
            # How confident was the model in the wrong direction?
            # Use the probability assigned to the wrong predicted class
            wrong_probs = probs[wrong_mask].gather(1, hard_preds[wrong_mask].unsqueeze(1)).squeeze(1)
            # Penalize proportional to confidence in wrong answer
            wrong_penalty = wrong_probs.mean()
        else:
            wrong_penalty = torch.zeros(1, device=direction_logits.device)

        total = self.cls_weight * cls_loss + self.wrong_trade_penalty * wrong_penalty

        # NaN guard
        if torch.isnan(total):
            if not hasattr(self, '_nan_warned'):
                print(f"  WARNING: NaN in loss — cls={cls_loss.item():.4f}")
                self._nan_warned = True
            total = cls_loss if not torch.isnan(cls_loss) else torch.zeros(1, device=direction_logits.device, requires_grad=True)

        # --- Metrics (for reporting only, not in loss) ---
        accuracy = (hard_preds == true_labels).float().mean().item()

        # Trade-only accuracy = win rate proxy
        if trade_mask.any():
            trade_accuracy = (hard_preds[trade_mask] == true_labels[trade_mask]).float().mean().item()
        else:
            trade_accuracy = 0.0

        # P&L metric (reporting only)
        rr_ratio = self.tp_points / self.sl_points
        true_onehot = F.one_hot(true_labels, num_classes=3).float()
        soft_correct = (probs * true_onehot).sum(dim=1)
        expected_pnl = soft_correct * rr_ratio - (1.0 - soft_correct) * 1.0

        # Sortino metric (reporting only)
        soft_pnl = expected_pnl
        downside = torch.clamp(soft_pnl, max=0)
        downside_std = torch.sqrt((downside ** 2).mean() + 1e-3)
        sortino_val = (soft_pnl.mean() / downside_std).clamp(-10, 10).item()

        # Confidence on trades
        conf_sig = torch.sigmoid(confidence).float()
        if trade_mask.any():
            avg_trade_conf = conf_sig[trade_mask].mean().item()
        else:
            avg_trade_conf = 0.0

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": 0.0,
            "sortino": sortino_val,
            "conf_loss": 0.0,
            "pnl": expected_pnl.mean().item(),
            "flat_rate": p_flat_mean.item(),
            "accuracy": accuracy,
            "trade_acc": trade_accuracy,
            "avg_trade_conf": avg_trade_conf,
            "wrong_penalty": wrong_penalty.item() if torch.is_tensor(wrong_penalty) else 0.0,
        }
        return total, metrics
