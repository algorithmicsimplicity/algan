"""Import-time configuration.

These values affect process/runtime initialisation and intentionally have no
public Python settings object. Set their environment variables before
importing :mod:`algan`.

The **render** device is the exception, and it lives here only because its
default is one of these environment reads. ``ALGAN_RENDER_DEVICE`` seeds
``SETTINGS.computing.render_device``, which is the runtime source of truth from
then on; :func:`render_device` is how engine code asks for it. Nothing in the
package may bind that value at import time -- ``from ... import _RENDER_DEVICE``
would capture a device the user can still change, which is why that name no
longer exists.
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

#: Memoized answer of :func:`_cuda_is_usable`. The probe allocates on the
#: device, and since ``render_device`` became a settings field this runs on
#: every ``SETTINGS.computing.set`` -- ``dataclasses.replace`` re-runs
#: ``__post_init__``, which re-validates the device. Whether CUDA works does not
#: change within a process, so ask once.
_CUDA_USABLE = None


def _cuda_is_usable() -> bool:
    global _CUDA_USABLE
    if _CUDA_USABLE is not None:
        return _CUDA_USABLE
    if not torch.cuda.is_available():
        _CUDA_USABLE = False
        return False
    try:
        torch.zeros((1,), device="cuda") + 1
        _CUDA_USABLE = True
    except Exception:
        _CUDA_USABLE = False
    return _CUDA_USABLE


def _auto_render_device() -> torch.device:
    if _cuda_is_usable():
        return torch.device("cuda")
    mps = getattr(torch, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def coerce_device(value: str | torch.device, source: str) -> torch.device:
    """Validate ``value`` as a Torch device, resolving ``'auto'``.

    ``source`` names what supplied the value -- an environment variable, or a
    settings field -- so a rejection says where to fix it. Availability is
    checked here rather than at first use: a render that silently falls back to
    the CPU because CUDA was unusable is a twenty-minute mystery, and the same
    check has to run whether the device arrived from the environment at startup
    or from ``SETTINGS.computing.set(render_device=...)`` mid-script.
    """
    if isinstance(value, torch.device):
        device = value
    else:
        raw = str(value).strip().lower()
        if raw == "auto":
            return _auto_render_device()
        try:
            device = torch.device(raw)
        except Exception as exc:
            raise AlganConfigurationError(
                f"{source} must name a valid Torch device or 'auto'"
            ) from exc
    if device.type == "cuda" and not _cuda_is_usable():
        raise AlganConfigurationError(
            f"{source} requests CUDA, but no usable CUDA runtime is available"
        )
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None or not mps.is_available():
            raise AlganConfigurationError(
                f"{source} requests MPS, but MPS is unavailable"
            )
    return device


def _parse_device(env_name: str, default: str | torch.device) -> torch.device:
    return coerce_device(env_str(env_name, str(default)), env_name)


_ANIMATION_DEVICE = _parse_device("ALGAN_ANIMATION_DEVICE", "cpu")

#: What ``SETTINGS.computing.render_device`` starts at. Read once, here; every
#: later read goes through :func:`render_device` so a runtime change is seen.
_DEFAULT_RENDER_DEVICE = _parse_device("ALGAN_RENDER_DEVICE", "auto")


def render_device() -> torch.device:
    """The device this process renders on, right now.

    Call it; never bind the result at import time. The device is
    ``SETTINGS.computing.render_device`` and a script may change it between
    renders, so a module-level ``RENDER_DEVICE = render_device()`` reintroduces
    exactly the staleness this function exists to remove.

    The import is function-local because ``SETTINGS`` is assembled from
    sections that import this module; by the time anything asks for a device
    that cycle is long closed.
    """
    from algan.settings import SETTINGS

    return SETTINGS.computing.render_device


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
