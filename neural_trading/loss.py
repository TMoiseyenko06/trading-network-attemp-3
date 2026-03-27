"""Custom loss for fixed TP/SL trading regime.

The model predicts direction (LONG/SHORT/FLAT) + confidence.
TP/SL are hard-coded (30pt TP, 20pt SL → 1.5 R:R).

Target: fewer, higher-quality trades. Three loss components:
  1. Label-smoothed cross-entropy — calibrated direction prediction
  2. Wrong-trade penalty — extra cost for wrong LONG/SHORT calls
  3. Confidence calibration — train confidence head as a real accuracy
     predictor so it can be used as a quality filter in live trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Classification loss targeting high win rate on trades.

    Cross-entropy + wrong-trade penalty + confidence calibration.
    The confidence head learns to predict whether the trade will be
    correct, making it a reliable filter for live/backtest.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.0,
        sortino_weight: float = 0.0,
        confidence_weight: float = 0.3,
        pnl_weight: float = 0.0,
        frequency_weight: float = 3.0,
        tp_points: float = 30.0,
        sl_points: float = 20.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.05,
        wrong_trade_penalty: float = 1.5,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.confidence_weight = confidence_weight
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
        #    but gets the direction wrong. Increased weight (1.5) forces
        #    the model to prefer FLAT over uncertain directional calls.
        trade_mask = hard_preds != 2                          # predicted a direction
        wrong_mask = trade_mask & (hard_preds != true_labels) # but got it wrong

        if wrong_mask.any():
            wrong_probs = probs[wrong_mask].gather(1, hard_preds[wrong_mask].unsqueeze(1)).squeeze(1)
            wrong_penalty = wrong_probs.mean()
        else:
            wrong_penalty = torch.zeros(1, device=direction_logits.device)

        # 3. Confidence calibration — train the confidence head to predict
        #    whether the directional prediction is correct.
        #    Target: 1.0 if pred == true_label, 0.0 if pred != true_label.
        #    Only on non-FLAT predictions (confidence is meaningless for FLAT).
        conf_sig = torch.sigmoid(confidence)
        conf_loss = torch.zeros(1, device=direction_logits.device)

        if trade_mask.any():
            # Binary target: 1 = correct trade, 0 = wrong trade
            correct = (hard_preds[trade_mask] == true_labels[trade_mask]).float()
            conf_on_trades = conf_sig[trade_mask]
            # BCE loss: train confidence to match correctness
            conf_loss = F.binary_cross_entropy(
                conf_on_trades, correct, reduction="mean",
            )

        total = (self.cls_weight * cls_loss
                 + self.wrong_trade_penalty * wrong_penalty
                 + self.confidence_weight * conf_loss)

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
        if trade_mask.any():
            avg_trade_conf = conf_sig[trade_mask].mean().item()
        else:
            avg_trade_conf = 0.0

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": 0.0,
            "sortino": sortino_val,
            "conf_loss": conf_loss.item() if torch.is_tensor(conf_loss) else 0.0,
            "pnl": expected_pnl.mean().item(),
            "flat_rate": p_flat_mean.item(),
            "accuracy": accuracy,
            "trade_acc": trade_accuracy,
            "avg_trade_conf": avg_trade_conf,
            "wrong_penalty": wrong_penalty.item() if torch.is_tensor(wrong_penalty) else 0.0,
        }
        return total, metrics
