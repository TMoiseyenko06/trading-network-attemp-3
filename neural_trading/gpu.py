"""GPU detection, auto-scaling, and optimization utilities."""

import torch
import math
from dataclasses import dataclass


@dataclass
class GPUProfile:
    """Hardware profile used to auto-scale model and training parameters."""
    device: torch.device
    name: str
    total_memory_gb: float
    compute_capability: tuple[int, int]
    num_sms: int
    use_amp: bool  # automatic mixed precision
    use_tf32: bool
    use_compile: bool  # torch.compile
    batch_size: int
    num_workers: int
    model_scale: float  # multiplier for hidden dims
    gradient_accumulation_steps: int
    pin_memory: bool


def detect_gpu() -> GPUProfile:
    """Detect GPU capabilities and return an optimized profile."""
    if not torch.cuda.is_available():
        return GPUProfile(
            device=torch.device("cpu"),
            name="CPU",
            total_memory_gb=0,
            compute_capability=(0, 0),
            num_sms=0,
            use_amp=False,
            use_tf32=False,
            use_compile=False,
            batch_size=32,
            num_workers=2,
            model_scale=0.5,
            gradient_accumulation_steps=4,
            pin_memory=False,
        )

    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / (1024 ** 3)
    cc = (props.major, props.minor)
    sms = props.multi_processor_count

    # AMP available on compute capability >= 7.0 (Volta+)
    use_amp = cc >= (7, 0)
    # TF32 available on Ampere+ (8.0+)
    use_tf32 = cc >= (8, 0)
    # torch.compile works best on Ampere+
    use_compile = cc >= (8, 0) and hasattr(torch, "compile")

    # Scale batch size based on VRAM
    if mem_gb >= 40:      # A100-80GB tier — massive VRAM headroom
        batch_size = 4096
        model_scale = 2.0
        grad_accum = 1
        workers = 8
    elif mem_gb >= 20:    # 4090 / 3090 / A100-40GB tier
        batch_size = 2048
        model_scale = 2.0
        grad_accum = 1
        workers = 8
    elif mem_gb >= 10:    # 3080 / A4000 tier
        batch_size = 256
        model_scale = 1.5
        grad_accum = 1
        workers = 6
    elif mem_gb >= 6:     # 3060 / 2060 tier
        batch_size = 128
        model_scale = 1.0
        grad_accum = 2
        workers = 4
    else:                 # low-end
        batch_size = 64
        model_scale = 0.75
        grad_accum = 4
        workers = 2

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True

    return GPUProfile(
        device=torch.device("cuda:0"),
        name=props.name,
        total_memory_gb=round(mem_gb, 1),
        compute_capability=cc,
        num_sms=sms,
        use_amp=use_amp,
        use_tf32=use_tf32,
        use_compile=use_compile,
        batch_size=batch_size,
        num_workers=workers,
        model_scale=model_scale,
        gradient_accumulation_steps=grad_accum,
        pin_memory=True,
    )


def scale_dim(base: int, scale: float) -> int:
    """Scale a hidden dimension by GPU scale factor, keep it divisible by 8."""
    return max(8, int(math.ceil(base * scale / 8)) * 8)
