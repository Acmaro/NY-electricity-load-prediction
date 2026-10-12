import random

import numpy as np
import torch


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks cuda > mps > cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    """Return the autocast dtype, or None for full fp32.

    'auto' uses bf16 on GPUs that support it (Ampere+, e.g. 4070 Ti, A100, H100),
    fp16 on older CUDA GPUs and fp32 elsewhere.
    """
    if precision == "fp32" or device.type != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"unknown precision: {precision}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
