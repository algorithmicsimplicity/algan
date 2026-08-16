"""Import-time configuration.

These values affect process/runtime initialisation and intentionally have no
public Python settings object. Set their environment variables before
importing :mod:`algan`.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raise the CUDA driver's JIT cache ceiling before anything can initialise a
# CUDA context (the torch import below probes one). Algan's Taichi kernels JIT
# to several hundred MB of ComputeCache entries; at the driver's default cap
# the cache sits in permanent LRU eviction and every fresh process re-JITs
# multi-second kernel modules (~12s measured on the debug scene's variant set).
# ``setdefault`` so an explicit user value always wins.
os.environ.setdefault("CUDA_CACHE_MAXSIZE", "4294967296")

import torch

from algan.environment import env_flag, env_int, env_str
from algan.errors import AlganConfigurationError


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros((1,), device="cuda") + 1
        return True
    except Exception:
        return False


def _auto_render_device() -> torch.device:
    if _cuda_is_usable():
        return torch.device("cuda")
    mps = getattr(torch, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_device(env_name: str, default: str | torch.device) -> torch.device:
    raw = env_str(env_name, str(default)).strip().lower()
    if raw == "auto":
        return _auto_render_device()
    try:
        device = torch.device(raw)
    except Exception as exc:
        raise AlganConfigurationError(
            f"{env_name} must name a valid Torch device or 'auto'"
        ) from exc
    if device.type == "cuda" and not _cuda_is_usable():
        raise AlganConfigurationError(
            f"{env_name} requests CUDA, but no usable CUDA runtime is available"
        )
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None or not mps.is_available():
            raise AlganConfigurationError(
                f"{env_name} requests MPS, but MPS is unavailable"
            )
    return device


_ANIMATION_DEVICE = _parse_device("ALGAN_ANIMATION_DEVICE", "cpu")
_RENDER_DEVICE = _parse_device("ALGAN_RENDER_DEVICE", "auto")

_ALGAN_HOME = Path(env_str("ALGAN_HOME") or Path.home() / ".algan").expanduser()
_CACHE_DIRECTORY = Path(
    env_str("ALGAN_CACHE_DIR") or _ALGAN_HOME / "cache"
).expanduser()
_TAICHI_CACHE_DIRECTORY = Path(
    env_str("TI_OFFLINE_CACHE_FILE_PATH") or _CACHE_DIRECTORY / "taichi"
).expanduser()

# These are baked into Taichi kernels or runtime layout at first materialisation.
_SOFT_SHADOW_SAMPLES = max(2, env_int("ALGAN_SOFT_SHADOW_SAMPLES", 8))
_HDR_BUFFER_F16 = env_flag("ALGAN_HDR_BUFFER_F16", False)
