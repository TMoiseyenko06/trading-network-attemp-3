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
        6. Direct P&L maximization: expected profit with soft confidence gate
        7. Selectivity: heavy penalty for overtrading (target 95% flat)
        8. Low-conviction penalty: per-sample cost for trading without confidence
    """

    def __init__(
        self,
        cls_weight: float = 0.4,
        mag_weight: float = 0.1,
        sortino_weight: float = 0.6,
        confidence_weight: float = 0.2,
        tp_sl_weight: float = 0.3,
        pnl_weight: float = 1.5,
        selectivity_weight: float = 15.0,
        low_conviction_weight: float = 3.0,
        conf_threshold: float = 0.7,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mag_weight = mag_weight
        self.sortino_weight = sortino_weight
        self.confidence_weight = confidence_weight
        self.tp_sl_weight = tp_sl_weight
        self.pnl_weight = pnl_weight
        self.selectivity_weight = selectivity_weight
        self.low_conviction_weight = low_conviction_weight
        self.conf_threshold = conf_threshold
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

        # 6. Direct P&L maximization — quality-gated profit objective
        #    Uses TRUE TP/SL so the model can't game profit by inflating TP / shrinking SL.
        #    Soft confidence gate lets PnL signal through for learning direction quality.
        #    Selectivity (#7) and low-conviction penalty (#8) handle trade filtering.
        p_flat = probs[:, 2]            # (B,) per-sample flat probability
        p_trade = 1.0 - p_flat          # how much the model wants to trade this bar
        conf_sig = torch.sigmoid(confidence).float()
        conf_gate = conf_sig            # soft gate — preserves learning signal
        expected_pnl = soft_correct * true_tp.float() - (1.0 - soft_correct) * true_sl.float()
        # Double-gated: must predict trade AND be confident to get reward
        gated_pnl = expected_pnl * p_trade * conf_gate
        pnl_loss = -gated_pnl.mean()

        # 7. Selectivity — squared penalty for overtrading, grows fast when far from target
        #    Target ~95% flat = ~20 trades per 390-bar session (~5 round trips)
        p_flat_mean = p_flat.mean()
        target_flat_rate = 0.95
        # Squared so penalty grows quadratically — 0.7 gap costs 4x more than 0.35 gap
        overtrading_penalty = F.relu(target_flat_rate - p_flat_mean) ** 2
        # Mild linear penalty for never trading
        undertrading_penalty = F.relu(p_flat_mean - 0.995) * 2.0
        selectivity_loss = overtrading_penalty + undertrading_penalty

        # 8. Low-conviction trade penalty — per-sample penalty for trading without
        #    sufficient confidence. Directly punishes each bar where model wants to
        #    trade (p_trade high) but confidence is below threshold.
        low_conf_mask = (self.conf_threshold - conf_sig).clamp(min=0)  # >0 when under threshold
        low_conviction_loss = (p_trade * low_conf_mask).mean()

        total = (
            self.cls_weight * cls_loss
            + self.mag_weight * mag_loss
            + self.sortino_weight * sortino
            + self.confidence_weight * conf_loss
            + self.tp_sl_weight * tp_sl_loss
            + self.pnl_weight * pnl_loss
            + self.selectivity_weight * selectivity_loss
            + self.low_conviction_weight * low_conviction_loss
        )

        # Accuracy metric (hard, for reporting only — only on non-flat predictions)
        hard_preds = direction_logits.argmax(dim=1)
        accuracy = (hard_preds == true_labels).float().mean().item()

        # Trade-only accuracy: how accurate when the model chooses to trade
        trade_mask = hard_preds != 2  # not FLAT
        if trade_mask.any():
            trade_accuracy = (hard_preds[trade_mask] == true_labels[trade_mask]).float().mean().item()
        else:
            trade_accuracy = 0.0

        # Metrics for monitoring
        rr_ratio = pred_tp / (pred_sl + 1e-8)
        avg_pnl = gated_pnl.mean().item()
        flat_rate = p_flat_mean.item()

        # Average confidence on trades (where model chose to trade)
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
            "flat_rate": flat_rate,
            "rr_ratio": rr_ratio.mean().item(),
            "accuracy": accuracy,
            "trade_acc": trade_accuracy,
            "avg_trade_conf": avg_trade_conf,
            "low_conv_loss": low_conviction_loss.item(),
        }
        return total, metrics
