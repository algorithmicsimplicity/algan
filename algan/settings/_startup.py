"""Import-time configuration.

These values affect process/runtime initialisation and intentionally have no
public Python settings object. Set their environment variables before
importing :mod:`algan`.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

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
    raw = os.environ.get(env_name, str(default)).strip().lower()
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

_ALGAN_HOME = Path(os.environ.get("ALGAN_HOME", Path.home() / ".algan")).expanduser()
_CACHE_DIRECTORY = Path(
    os.environ.get("ALGAN_CACHE_DIR", _ALGAN_HOME / "cache")
).expanduser()
_TAICHI_CACHE_DIRECTORY = Path(
    os.environ.get("TI_OFFLINE_CACHE_FILE_PATH", _CACHE_DIRECTORY / "taichi")
).expanduser()

# These are baked into Taichi kernels or runtime layout at first materialisation.
_SOFT_SHADOW_SAMPLES = max(2, int(os.environ.get("ALGAN_SOFT_SHADOW_SAMPLES", "8")))
_HDR_BUFFER_F16 = os.environ.get("ALGAN_HDR_BUFFER_F16", "0") == "1"
