"""Custom loss for fixed TP/SL trading regime.

The model only predicts direction (LONG/SHORT/FLAT) + confidence.
TP/SL are hard-coded (35pt TP, 20pt SL). With 55%+ win rate, the
fixed R:R of 1.75 guarantees profit factor > 2.0.

Loss = cross-entropy only. Trade frequency is handled by the risk
manager (confidence threshold, max trades/day, bar gap). The loss
should focus 100% on getting direction right.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Pure classification loss for fixed TP/SL regime.

    Just cross-entropy. The model learns direction accuracy (= win rate),
    the risk manager handles trade frequency. No competing objectives.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.0,
        sortino_weight: float = 0.0,
        confidence_weight: float = 0.0,
        pnl_weight: float = 0.0,
        frequency_weight: float = 3.0,
        tp_points: float = 35.0,
        sl_points: float = 20.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.frequency_weight = frequency_weight
        self.tp_points = tp_points
        self.sl_points = sl_points
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

        # Classification loss — the ONLY objective
        # Higher accuracy = higher win rate = more profit with fixed R:R
        cls_loss = F.cross_entropy(direction_logits, true_labels, weight=self.class_weights)

        # Probabilities for metrics
        probs = F.softmax(direction_logits, dim=1)  # (B, 3)
        p_flat_mean = probs[:, 2].mean()

        total = self.cls_weight * cls_loss

        # NaN guard
        if torch.isnan(total):
            if not hasattr(self, '_nan_warned'):
                print(f"  WARNING: NaN in loss — cls={cls_loss.item():.4f}")
                self._nan_warned = True
            total = cls_loss if not torch.isnan(cls_loss) else torch.zeros(1, device=direction_logits.device, requires_grad=True)

        # --- Metrics (for reporting only, not in loss) ---
        hard_preds = direction_logits.argmax(dim=1)
        accuracy = (hard_preds == true_labels).float().mean().item()

        # Trade-only accuracy = win rate proxy
        trade_mask = hard_preds != 2  # not FLAT
        if trade_mask.any():
            trade_accuracy = (hard_preds[trade_mask] == true_labels[trade_mask]).float().mean().item()
        else:
            trade_accuracy = 0.0

        # P&L metric (reporting only) — what the fixed R:R would yield
        rr_ratio = self.tp_points / self.sl_points  # 1.75
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
        }
        return total, metrics
