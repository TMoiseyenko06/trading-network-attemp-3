"""
GPU detection and auto-scaling configuration.

Profiles the available CUDA device and returns model/training hyperparameters
scaled to the GPU's memory and compute capability so the network automatically
uses as much hardware as available without OOM errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUProfile:
    # Hardware
    device: torch.device
    gpu_name: str
    vram_gb: float
    compute_cap: int          # e.g. 86 for Ampere RTX 3090, 89 for Ada L4
    num_gpus: int
    scale_tier: str           # "cpu" | "small" | "medium" | "large" | "xl"

    # Training knobs (all scaled to GPU)
    batch_size: int
    grad_accum_steps: int     # effective batch = batch_size * grad_accum_steps

    # Architecture knobs
    conv_filters: int         # base channel count for causal conv stage
    num_dilation_stacks: int  # how many DilatedConvBlock stacks
    attn_d_model: int         # transformer d_model
    attn_nhead: int
    attn_layers: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float

    # Precision / compiler
    use_amp: bool
    amp_dtype: Optional[torch.dtype]
    use_compile: bool         # torch.compile (PyTorch 2.x, Ampere+)
    use_channels_last: bool   # memory format optimisation for conv-heavy nets

    # DataLoader
    num_workers: int
    pin_memory: bool
    prefetch_factor: int


def get_gpu_profile(force_cpu: bool = False) -> GPUProfile:
    """
    Detect the best available CUDA device and return a fully populated
    GPUProfile with hyperparameters scaled to the hardware.
    """
    if force_cpu or not torch.cuda.is_available():
        logger.warning("CUDA unavailable – running on CPU (slow).")
        return _cpu_profile()

    # Pick the GPU with the most VRAM if multiple are present
    best_idx = 0
    best_vram = 0
    for i in range(torch.cuda.device_count()):
        vram = torch.cuda.get_device_properties(i).total_memory
        if vram > best_vram:
            best_vram = vram
            best_idx = i

    torch.cuda.set_device(best_idx)
    device = torch.device(f"cuda:{best_idx}")
    props = torch.cuda.get_device_properties(best_idx)

    vram_gb: float = props.total_memory / (1024 ** 3)
    compute_cap: int = props.major * 10 + props.minor
    num_gpus: int = torch.cuda.device_count()

    # ---------- scale tier -------------------------------------------------
    if vram_gb >= 60:
        tier = "xl"
    elif vram_gb >= 24:
        tier = "large"
    elif vram_gb >= 10:
        tier = "medium"
    else:
        tier = "small"

    params = _tier_params(tier)

    # ---------- precision ---------------------------------------------------
    # BF16 preferred on Ampere (SM 80+) – numerically more stable than FP16
    # FP16 on Volta / Turing (SM 70-79)
    # No AMP below SM 70
    if compute_cap >= 80:
        use_amp = True
        amp_dtype = torch.bfloat16
    elif compute_cap >= 70:
        use_amp = True
        amp_dtype = torch.float16
    else:
        use_amp = False
        amp_dtype = None

    # torch.compile requires PyTorch 2.x and benefits from Ampere+
    use_compile = hasattr(torch, "compile") and compute_cap >= 80

    # channels_last is beneficial for conv-heavy models on CUDA
    use_channels_last = torch.cuda.is_available()

    # DataLoader workers: 2 per GPU, capped at 8
    num_workers = min(8, 2 * num_gpus)

    profile = GPUProfile(
        device=device,
        gpu_name=props.name,
        vram_gb=round(vram_gb, 1),
        compute_cap=compute_cap,
        num_gpus=num_gpus,
        scale_tier=tier,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_compile=use_compile,
        use_channels_last=use_channels_last,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        **params,
    )

    _log_profile(profile)
    return profile


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tier_params(tier: str) -> dict:
    """Return architecture/training params for a given scale tier."""
    tiers = {
        "xl": dict(
            batch_size=512,
            grad_accum_steps=1,
            conv_filters=256,
            num_dilation_stacks=4,
            attn_d_model=256,
            attn_nhead=16,
            attn_layers=4,
            lstm_hidden=512,
            lstm_layers=3,
            dropout=0.15,
        ),
        "large": dict(
            batch_size=256,
            grad_accum_steps=1,
            conv_filters=128,
            num_dilation_stacks=3,
            attn_d_model=128,
            attn_nhead=8,
            attn_layers=3,
            lstm_hidden=256,
            lstm_layers=2,
            dropout=0.1,
        ),
        "medium": dict(
            batch_size=128,
            grad_accum_steps=2,
            conv_filters=64,
            num_dilation_stacks=3,
            attn_d_model=64,
            attn_nhead=4,
            attn_layers=2,
            lstm_hidden=128,
            lstm_layers=2,
            dropout=0.1,
        ),
        "small": dict(
            batch_size=64,
            grad_accum_steps=4,
            conv_filters=32,
            num_dilation_stacks=2,
            attn_d_model=32,
            attn_nhead=2,
            attn_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.05,
        ),
    }
    return tiers[tier]


def _cpu_profile() -> GPUProfile:
    return GPUProfile(
        device=torch.device("cpu"),
        gpu_name="CPU",
        vram_gb=0.0,
        compute_cap=0,
        num_gpus=0,
        scale_tier="cpu",
        batch_size=32,
        grad_accum_steps=4,
        conv_filters=32,
        num_dilation_stacks=2,
        attn_d_model=32,
        attn_nhead=2,
        attn_layers=2,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.05,
        use_amp=False,
        amp_dtype=None,
        use_compile=False,
        use_channels_last=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
    )


def _log_profile(p: GPUProfile) -> None:
    logger.info("=" * 60)
    logger.info("GPU PROFILE")
    logger.info(f"  Device       : {p.gpu_name} ({p.vram_gb:.1f} GB VRAM)")
    logger.info(f"  Compute cap  : SM {p.compute_cap}")
    logger.info(f"  Num GPUs     : {p.num_gpus}")
    logger.info(f"  Scale tier   : {p.scale_tier.upper()}")
    logger.info(f"  Precision    : {'BF16' if p.amp_dtype == torch.bfloat16 else 'FP16' if p.amp_dtype == torch.float16 else 'FP32'}")
    logger.info(f"  torch.compile: {p.use_compile}")
    logger.info(f"  Batch size   : {p.batch_size} (× {p.grad_accum_steps} grad accum = {p.batch_size * p.grad_accum_steps} effective)")
    logger.info(f"  Conv filters : {p.conv_filters}")
    logger.info(f"  LSTM hidden  : {p.lstm_hidden}")
    logger.info(f"  Attn d_model : {p.attn_d_model} ({p.attn_nhead} heads, {p.attn_layers} layers)")
    logger.info("=" * 60)
