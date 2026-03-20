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
        7. Trade selectivity: penalizes model for trading too often (target ~60% flat)
        8. R:R incentive: rewards predicted TP/SL ratio above minimum threshold
    """

    def __init__(
        self,
        cls_weight: float = 0.8,
        mag_weight: float = 0.2,
        sortino_weight: float = 0.2,
        confidence_weight: float = 0.2,
        tp_sl_weight: float = 0.6,
        pnl_weight: float = 1.0,
        frequency_weight: float = 0.3,
        rr_weight: float = 0.4,
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
        self.rr_weight = rr_weight
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

        # Force float32 for all loss math — float16 under AMP causes NaN
        # due to underflow in small TP/SL values and division precision loss
        direction_logits = direction_logits.float()
        confidence = confidence.float()
        pred_magnitude = pred_magnitude.float()
        pred_tp = pred_tp.float()
        pred_sl = pred_sl.float()
        true_magnitude = true_magnitude.float()
        true_tp = true_tp.float()
        true_sl = true_sl.float()

        # 1. Classification loss (with class weights to handle imbalance)
        cls_loss = F.cross_entropy(direction_logits, true_labels, weight=self.class_weights)

        # 2. Magnitude regression loss
        mag_loss = F.mse_loss(pred_magnitude, true_magnitude)

        # Shared quantities
        probs = F.softmax(direction_logits, dim=1)  # (B, 3)
        true_onehot = F.one_hot(true_labels, num_classes=direction_logits.shape[1]).float()
        soft_correct = (probs * true_onehot).sum(dim=1)  # (B,) in [0, 1]

        # 3. Sortino ratio (detached TP/SL — only teaches direction quality)
        tp_detached = pred_tp.detach()
        sl_detached = pred_sl.detach()
        soft_pnl_sortino = soft_correct * tp_detached - (1.0 - soft_correct) * sl_detached
        downside = torch.clamp(soft_pnl_sortino, max=0)
        downside_var = (downside ** 2).mean()
        downside_std = torch.sqrt(downside_var + 1e-3)
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
        #    Uses TRUE TP/SL so the model can't game profit by inflating predicted TP.
        #    Gradient flows through direction only (not TP/SL sizing).
        #    TP/SL learning happens via the Huber loss (#5) instead.
        p_trade = 1.0 - probs[:, 2]  # probability of NOT predicting flat
        expected_pnl = soft_correct * true_tp - (1.0 - soft_correct) * true_sl
        # Scale by trade probability: no reward for being right if you don't trade
        weighted_pnl = expected_pnl * p_trade
        pnl_loss = -weighted_pnl.mean()  # negative because we maximize profit

        # 7. Trade selectivity — penalize trading too often
        #    Target: model should predict FLAT ~60% of time (selective trading)
        #    Penalize both over-trading (flat < 40%) and never-trading (flat > 80%)
        p_flat_mean = probs[:, 2].mean()
        selectivity_penalty = F.relu(0.4 - p_flat_mean) + F.relu(p_flat_mean - 0.8)

        # 8. R:R incentive — reward predicted TP/SL ratio above 1.5
        #    Detached: gradient through tp/sl division explodes when sl is tiny at init.
        #    TP/SL sizing is learned via Huber loss (#5); this only shapes the ratio.
        rr_pred = pred_tp.detach() / (pred_sl.detach() + 1e-6)
        rr_penalty = F.relu(1.5 - rr_pred).mean()

        total = (
            self.cls_weight * cls_loss
            + self.mag_weight * mag_loss
            + self.sortino_weight * sortino
            + self.confidence_weight * conf_loss
            + self.tp_sl_weight * tp_sl_loss
            + self.pnl_weight * pnl_loss
            + self.frequency_weight * selectivity_penalty
            + self.rr_weight * rr_penalty
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

        # Metrics for monitoring
        rr_ratio = pred_tp / (pred_sl + 1e-6)
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
            "tp_sl_loss": tp_sl_loss.item(),
            "pnl": avg_pnl,
            "flat_rate": p_flat_mean.item(),
            "rr_ratio": rr_ratio.mean().item(),
            "rr_penalty": rr_penalty.item(),
            "accuracy": accuracy,
            "trade_acc": trade_accuracy,
            "avg_trade_conf": avg_trade_conf,
        }
        return total, metrics
