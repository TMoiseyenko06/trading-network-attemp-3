"""
3-stage neural network for NQ scalping from raw OHLCV.

Stage 1 — Feature Learning   : Parallel causal dilated convolutions
Stage 2 — Context & Memory   : Multi-head self-attention → LSTM
Stage 3 — Decision Head      : direction (3-class), confidence (0-1), magnitude

All layer sizes are driven by the GPUProfile so the model auto-scales
to the available hardware.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPUProfile


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D convolution with left-only (causal) padding — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class DilatedConvBlock(nn.Module):
    """
    Parallel causal convolutions at multiple dilation rates, concatenated
    then projected back down.  Learns short-term *and* long-term patterns
    in a single layer — dilation=1 ≈ 3-bar pattern, dilation=16 ≈ 48-bar.
    """

    def __init__(self, in_ch: int, out_ch: int, dilations: list[int], dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            CausalConv1d(in_ch, out_ch, kernel_size=3, dilation=d)
            for d in dilations
        ])
        concat_ch = out_ch * len(dilations)
        self.bn = nn.BatchNorm1d(concat_ch)
        self.proj = nn.Conv1d(concat_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut when dimensions match
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.proj(out)
        out = self.dropout(out)
        return out + identity


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TradingNetwork(nn.Module):
    """
    End-to-end OHLCV → (direction, confidence, magnitude) network.

    Input : (B, T, F)  — T timesteps, F features per bar
    Output: direction logits (B,3), confidence (B,1), magnitude (B,1)
    """

    def __init__(self, num_features: int, profile: GPUProfile):
        super().__init__()
        cf = profile.conv_filters
        d_model = profile.attn_d_model
        nhead = profile.attn_nhead
        n_attn = profile.attn_layers
        lh = profile.lstm_hidden
        n_lstm = profile.lstm_layers
        drop = profile.dropout

        # --- Stage 1: Causal dilated convolution stack ----------------------
        dilation_sets = [
            [1, 2, 4, 8],
            [1, 4, 8, 16],
            [1, 8, 16, 32],
            [1, 16, 32, 64],
        ][: profile.num_dilation_stacks]

        conv_layers: list[nn.Module] = []
        in_ch = num_features
        for dilations in dilation_sets:
            conv_layers.append(DilatedConvBlock(in_ch, cf, dilations, dropout=drop))
            in_ch = cf
            cf = min(cf * 2, profile.conv_filters * 4)   # grow channels gradually

        self.conv_stage = nn.Sequential(*conv_layers)
        self.conv_out_ch = in_ch   # channels after last block

        # Project conv output to attention d_model
        self.conv_to_attn = nn.Linear(self.conv_out_ch, d_model)

        # --- Stage 2: Multi-head self-attention + LSTM ----------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN for more stable training
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=n_attn)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lh,
            num_layers=n_lstm,
            batch_first=True,
            dropout=drop if n_lstm > 1 else 0.0,
        )
        self.lstm_ln = nn.LayerNorm(lh)

        # --- Stage 3: Decision heads ----------------------------------------
        self.direction_head = nn.Sequential(
            nn.Linear(lh, lh // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(lh // 2, 3),       # logits for [SL, TP, Timeout]
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(lh, lh // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(lh // 2, 1),
            nn.Sigmoid(),
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(lh, lh // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(lh // 2, 1),
        )

        self._init_weights()

    # ---- weight init -------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.kaiming_normal_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 for better long-term memory
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    # ---- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, F) float tensor — T bars, F features per bar

        Returns
        -------
        direction : (B, 3)  logits
        confidence: (B, 1)  probability 0-1
        magnitude : (B, 1)  expected move (signed)
        """
        # Stage 1: conv expects (B, C, T)
        h = x.permute(0, 2, 1)
        h = self.conv_stage(h)
        # Back to (B, T, C)
        h = h.permute(0, 2, 1)

        # Project to attention dimension
        h = self.conv_to_attn(h)

        # Stage 2: attention → LSTM
        # Build causal mask so attention can't peek forward
        T = h.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = self.attention(h, mask=causal_mask)

        h, _ = self.lstm(h)
        h = self.lstm_ln(h[:, -1, :])   # take last timestep

        # Stage 3: heads
        direction = self.direction_head(h)
        confidence = self.confidence_head(h)
        magnitude = self.magnitude_head(h)

        return direction, confidence, magnitude


def build_model(num_features: int, profile: GPUProfile) -> TradingNetwork:
    """Construct, optimise, and move the model to the correct device."""
    model = TradingNetwork(num_features, profile)
    model = model.to(profile.device)

    if profile.use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if profile.use_compile:
        model = torch.compile(model, mode="reduce-overhead")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[model] {param_count:,} parameters — tier={profile.scale_tier}")
    return model
