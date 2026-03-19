"""Three-stage neural network: Causal Conv -> Attention+LSTM -> Decision Head."""

import torch
import torch.nn as nn
import math


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class MultiScaleConvBlock(nn.Module):
    """Parallel causal convolutions at multiple dilation rates."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(in_ch, out_ch, kernel_size, d),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            )
            for d in dilations
        ])
        # Project concatenated branches back to out_ch
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch * len(dilations), out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.proj(torch.cat(outs, dim=1))


class FeatureLearner(nn.Module):
    """Stage 1 — stacked multi-scale causal convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            MultiScaleConvBlock(
                hidden_dim, hidden_dim,
                kernel_size=3,
                dilations=[1, 2, 4, 8],
            )
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = self.dropout(x)
        # Back to (batch, seq_len, hidden_dim)
        return x.transpose(1, 2)


class ContextMemory(nn.Module):
    """Stage 2 — multi-head self-attention + LSTM."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, lstm_layers: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (causal — no future leakage)
        seq_len = x.size(1)
        # Cache the causal mask to avoid regenerating every forward pass
        if not hasattr(self, "_causal_mask") or self._causal_mask.size(0) != seq_len or self._causal_mask.device != x.device:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device,
            )
        mask = self._causal_mask
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, is_causal=True)
        x = x + self.dropout(attn_out)

        # LSTM with residual
        normed = self.norm2(x)
        lstm_out, _ = self.lstm(normed)
        x = x + self.dropout(lstm_out)

        return x


class DecisionHead(nn.Module):
    """Stage 3 — five simultaneous outputs from the final hidden state.

    TP/SL bounds prevent degenerate strategies:
      - min_tp_pct: floor on take-profit so the model can't collect pennies
      - max_tp_pct: ceiling so TP stays realistic
      - min_sl_pct: floor on stop-loss (can't be tighter than noise)
      - max_sl_pct: ceiling so SL can't be "never hit"
    """

    # Bounds in raw pct space (training targets are clipped to match)
    MIN_TP_PCT = 0.001    # ~24 NQ pts at 24000 — minimum meaningful TP
    MAX_TP_PCT = 0.015    # ~360 NQ pts — reasonable upper bound
    MIN_SL_PCT = 0.0005   # ~12 NQ pts — can't be tighter than noise
    MAX_SL_PCT = 0.008    # ~192 NQ pts — prevents "never hit" SL

    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.direction = nn.Linear(hidden_dim, num_classes)   # softmax over long/short/flat
        self.confidence = nn.Linear(hidden_dim, 1)            # sigmoid -> [0, 1]
        self.magnitude = nn.Linear(hidden_dim, 1)             # expected move size
        self.tp_head = nn.Linear(hidden_dim, 1)               # take-profit pct
        self.sl_head = nn.Linear(hidden_dim, 1)               # stop-loss pct

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, hidden_dim) — take last timestep
        h = self.shared(x[:, -1, :])
        direction = self.direction(h)              # raw logits
        confidence = self.confidence(h).squeeze(-1)  # raw logit; sigmoid applied in loss/inference
        magnitude = torch.relu(self.magnitude(h)).squeeze(-1)

        # TP/SL via sigmoid scaled to [min, max] range — prevents degenerate
        # strategies where TP ≈ 0 (penny picking) or SL → ∞ (never hit)
        tp_raw = torch.sigmoid(self.tp_head(h)).squeeze(-1)  # (0, 1)
        sl_raw = torch.sigmoid(self.sl_head(h)).squeeze(-1)  # (0, 1)
        pred_tp = self.MIN_TP_PCT + tp_raw * (self.MAX_TP_PCT - self.MIN_TP_PCT)
        pred_sl = self.MIN_SL_PCT + sl_raw * (self.MAX_SL_PCT - self.MIN_SL_PCT)

        return direction, confidence, magnitude, pred_tp, pred_sl


class NeuralOHLCVNet(nn.Module):
    """Full 3-stage network: raw OHLCV features -> trade decisions."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        conv_blocks: int = 3,
        attn_heads: int = 4,
        lstm_layers: int = 2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.feature_learner = FeatureLearner(input_dim, hidden_dim, conv_blocks)
        self.context_memory = ContextMemory(hidden_dim, attn_heads, lstm_layers)
        self.decision_head = DecisionHead(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, lookback, input_dim)
        Returns:
            direction_logits: (batch, num_classes)
            confidence: (batch,)
            magnitude: (batch,)
            pred_tp: (batch,) predicted take-profit pct
            pred_sl: (batch,) predicted stop-loss pct
        """
        features = self.feature_learner(x)
        context = self.context_memory(features)
        return self.decision_head(context)
