"""Custom loss combining classification, profit maximization, and auxiliary objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Multi-objective loss for the trading network.

    Components:
        1. Cross-entropy for direction classification (with optional class weights)
        2. MSE for magnitude prediction
        3. Sortino ratio using detached TP/SL (teaches direction quality)
        4. Confidence calibration via soft correctness probabilities
        5. TP/SL regression: Huber loss on predicted take-profit and stop-loss
        6. Direct P&L maximization: expected profit with full gradient through TP/SL
        7. Trade frequency bonus: penalizes the model for predicting FLAT too often
    """

    def __init__(
        self,
        cls_weight: float = 0.8,
        mag_weight: float = 0.2,
        sortino_weight: float = 0.2,
        confidence_weight: float = 0.2,
        tp_sl_weight: float = 0.3,
        pnl_weight: float = 1.0,
        frequency_weight: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.sortino_weight = sortino_weight
        self.confidence_weight = confidence_weight
        self.tp_sl_weight = tp_sl_weight
        self.pnl_weight = pnl_weight
        self.frequency_weight = frequency_weight
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

        # Shared quantities
        probs = F.softmax(direction_logits.float(), dim=1)  # (B, 3)
        true_onehot = F.one_hot(true_labels, num_classes=direction_logits.shape[1]).float()
        soft_correct = (probs * true_onehot).sum(dim=1)  # (B,) in [0, 1]

        # 3. Sortino ratio (detached TP/SL — only teaches direction quality)
        tp_detached = pred_tp.detach().float()
        sl_detached = pred_sl.detach().float()
        soft_pnl_sortino = soft_correct * tp_detached - (1.0 - soft_correct) * sl_detached
        downside = torch.clamp(soft_pnl_sortino, max=0)
        downside_var = (downside ** 2).mean()
        downside_std = torch.sqrt(downside_var + 1e-4)
        sortino = -(soft_pnl_sortino.mean() / downside_std).clamp(-10, 10)

        # 4. Confidence calibration (soft correctness as target)
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence, soft_correct.detach(),
        )

        # 5. TP/SL regression (Huber for robustness to outliers)
        tp_loss = F.huber_loss(pred_tp, true_tp, delta=0.01)
        sl_loss = F.huber_loss(pred_sl, true_sl, delta=0.01)
        tp_sl_loss = tp_loss + sl_loss

        # 6. Direct P&L maximization — the core profit objective
        #    Expected PnL = P(correct) * TP - P(wrong) * SL
        #    Full gradient flows through both direction AND TP/SL predictions.
        #    The model learns to jointly optimize: pick good entries AND size TP/SL well.
        #
        #    We weight by trade probability (1 - P(flat)) so the model is incentivized
        #    to actually take trades, not hide in FLAT to avoid losses.
        p_trade = 1.0 - probs[:, 2]  # probability of NOT predicting flat
        expected_pnl = soft_correct * pred_tp.float() - (1.0 - soft_correct) * pred_sl.float()
        # Scale by trade probability: no reward for being right if you don't trade
        weighted_pnl = expected_pnl * p_trade
        pnl_loss = -weighted_pnl.mean()  # negative because we maximize profit

        # 7. Trade frequency bonus — penalize excessive FLAT predictions
        #    Target: model should predict FLAT ≤ ~20% of the time
        #    Penalty kicks in when P(flat) > target_flat_rate
        p_flat_mean = probs[:, 2].mean()
        target_flat_rate = 0.2
        frequency_penalty = F.relu(p_flat_mean - target_flat_rate)

        total = (
            self.cls_weight * cls_loss
            + self.mag_weight * mag_loss
            + self.sortino_weight * sortino
            + self.confidence_weight * conf_loss
            + self.tp_sl_weight * tp_sl_loss
            + self.pnl_weight * pnl_loss
            + self.frequency_weight * frequency_penalty
        )

        # Accuracy metric (hard, for reporting only)
        hard_preds = direction_logits.argmax(dim=1)
        accuracy = (hard_preds == true_labels).float().mean().item()

        # Metrics for monitoring
        rr_ratio = pred_tp / (pred_sl + 1e-8)
        avg_pnl = expected_pnl.mean().item()
        flat_rate = p_flat_mean.item()

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": mag_loss.item(),
            "sortino": -sortino.item(),
            "conf_loss": conf_loss.item(),
            "tp_sl_loss": tp_sl_loss.item(),
            "pnl": avg_pnl,
            "flat_rate": flat_rate,
            "rr_ratio": rr_ratio.mean().item(),
            "accuracy": accuracy,
        }
        return total, metrics
