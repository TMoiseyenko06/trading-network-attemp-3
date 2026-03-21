"""Custom loss combining classification, profit maximization, and auxiliary objectives.

Designed for fixed TP/SL regime: the model only predicts direction + confidence,
not TP/SL levels. TP/SL are hard-coded (e.g. 35pt TP, 20pt SL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TradingLoss(nn.Module):
    """Multi-objective loss for the trading network (fixed TP/SL regime).

    Components:
        1. Cross-entropy for direction classification (with optional class weights)
        2. MSE for magnitude prediction
        3. Sortino ratio using fixed TP/SL ratio (teaches direction quality)
        4. Confidence calibration via soft correctness probabilities
        5. Direct P&L maximization using fixed R:R ratio
        6. Trade selectivity: penalizes model for trading too often
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        mag_weight: float = 0.1,
        sortino_weight: float = 0.3,
        confidence_weight: float = 0.3,
        pnl_weight: float = 1.0,
        frequency_weight: float = 0.5,
        tp_points: float = 35.0,
        sl_points: float = 20.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.sortino_weight = sortino_weight
        self.confidence_weight = confidence_weight
        self.pnl_weight = pnl_weight
        self.frequency_weight = frequency_weight
        # Fixed R:R for P&L and Sortino calculations
        self.tp_points = tp_points
        self.sl_points = sl_points
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        confidence: torch.Tensor,          # (B,)
        pred_magnitude: torch.Tensor,      # (B,)
        pred_tp: torch.Tensor,             # (B,) — ignored (kept for model compat)
        pred_sl: torch.Tensor,             # (B,) — ignored (kept for model compat)
        true_labels: torch.Tensor,         # (B,) int
        true_magnitude: torch.Tensor,      # (B,) float
        true_tp: torch.Tensor,             # (B,) float — unused
        true_sl: torch.Tensor,             # (B,) float — unused
    ) -> tuple[torch.Tensor, dict[str, float]]:

        # Force float32 for all loss math — float16 under AMP causes NaN
        direction_logits = direction_logits.float()
        confidence = confidence.float()
        pred_magnitude = pred_magnitude.float()
        true_magnitude = true_magnitude.float()

        # 1. Classification loss (with class weights to handle imbalance)
        cls_loss = F.cross_entropy(direction_logits, true_labels, weight=self.class_weights)

        # 2. Magnitude regression loss
        mag_loss = F.mse_loss(pred_magnitude, true_magnitude)

        # Shared quantities
        probs = F.softmax(direction_logits, dim=1)  # (B, 3)
        true_onehot = F.one_hot(true_labels, num_classes=direction_logits.shape[1]).float()
        soft_correct = (probs * true_onehot).sum(dim=1)  # (B,) in [0, 1]

        # 3. Sortino ratio using fixed TP/SL ratio
        #    Measures risk-adjusted quality of direction predictions
        rr_ratio = self.tp_points / self.sl_points  # fixed 1.75 for 35/20
        soft_pnl = soft_correct * rr_ratio - (1.0 - soft_correct) * 1.0
        downside = torch.clamp(soft_pnl, max=0)
        downside_var = (downside ** 2).mean()
        downside_std = torch.sqrt(downside_var + 1e-3)
        sortino = -(soft_pnl.mean() / downside_std).clamp(-10, 10)

        # 4. Confidence calibration (soft correctness as target)
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence, soft_correct.detach(),
        )

        # 5. Direct P&L maximization using normalized R:R
        #    Uses ratio (not raw points) so values stay in [-1, +rr_ratio] range,
        #    preventing float16 overflow under AMP
        p_trade = 1.0 - probs[:, 2]  # probability of NOT predicting flat
        expected_pnl = soft_correct * rr_ratio - (1.0 - soft_correct) * 1.0
        weighted_pnl = expected_pnl * p_trade
        pnl_loss = -weighted_pnl.mean()

        # 6. Trade selectivity — model should be highly selective (1-5 trades/day)
        #    Target: flat 80-95% of the time (only trade high-conviction setups)
        p_flat_mean = probs[:, 2].mean()
        selectivity_penalty = F.relu(0.80 - p_flat_mean) + F.relu(p_flat_mean - 0.95)

        total = (
            self.cls_weight * cls_loss
            + self.mag_weight * mag_loss
            + self.sortino_weight * sortino
            + self.confidence_weight * conf_loss
            + self.pnl_weight * pnl_loss
            + self.frequency_weight * selectivity_penalty
        )

        # Accuracy metric (hard, for reporting only)
        hard_preds = direction_logits.argmax(dim=1)
        accuracy = (hard_preds == true_labels).float().mean().item()

        # Trade-only accuracy: how accurate when the model chooses to trade
        trade_mask = hard_preds != 2  # not FLAT
        if trade_mask.any():
            trade_accuracy = (hard_preds[trade_mask] == true_labels[trade_mask]).float().mean().item()
        else:
            trade_accuracy = 0.0

        avg_pnl = expected_pnl.mean().item()

        # Average confidence on trades (where model chose to trade)
        conf_sig = torch.sigmoid(confidence).float()
        trade_conf_mask = hard_preds != 2
        if trade_conf_mask.any():
            avg_trade_conf = conf_sig[trade_conf_mask].mean().item()
        else:
            avg_trade_conf = 0.0

        metrics = {
            "cls_loss": cls_loss.item(),
            "mag_loss": mag_loss.item(),
            "sortino": -sortino.item(),
            "conf_loss": conf_loss.item(),
            "pnl": avg_pnl,
            "flat_rate": p_flat_mean.item(),
            "accuracy": accuracy,
            "trade_acc": trade_accuracy,
            "avg_trade_conf": avg_trade_conf,
        }
        return total, metrics
