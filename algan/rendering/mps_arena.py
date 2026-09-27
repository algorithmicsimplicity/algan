"""Standalone Metal storage for managed render arenas.

Only managed, nonempty MPS arenas use this allocator. Ordinary tensors, dry-run
arenas and other devices retain Torch's allocator. The native DLPack owner lives
until the final storage view (including a Quadrants import) is released. No
Python finalizer, global tensor registry, CPU staging, or runtime compiler is
involved. The existing zero-copy launch wrapper still owns queue ordering.

Torch's live-allocation counter excludes externally owned DLPack storage. Read
``allocated_bytes`` alongside it for arena sizing; do NOT add it to the driver
counter, which already includes the native Metal resources.
"""

from __future__ import annotations

import importlib
import operator
import sys

import torch

MAX_ARENA_BYTES = (1 << 31) - (1 << 20)
_NATIVE = None


def _native():
    global _NATIVE
    if _NATIVE is None:
        if sys.platform != "darwin":
            raise RuntimeError("Standalone MPS arenas are available only on macOS")
        try:
            _NATIVE = importlib.import_module("algan.rendering._mps_arena_native")
        except ImportError as error:
            raise RuntimeError(
                "Algan's native MPS arena extension is missing or cannot be loaded. "
                "Install the macOS Algan wheel, or install Apple's Command Line "
                "Tools (xcode-select --install) and rebuild the source installation "
                "with python -m pip install --no-cache-dir --force-reinstall --no-deps . "
                "The unsafe heap allocator is not used as a fallback."
            ) from error
    return _NATIVE


def allocated_bytes() -> int:
    """Return live external storage bytes without initializing Metal."""
    return 0 if _NATIVE is None else int(_NATIVE.allocated_bytes())


def allocate(num_bytes: int) -> torch.Tensor:
    """Allocate a contiguous uint8 MPS tensor with native storage ownership."""
    if isinstance(num_bytes, bool):
        raise TypeError("MPS arena size must be an integer byte count, not bool")
    num_bytes = operator.index(num_bytes)
    if not 0 < num_bytes <= MAX_ARENA_BYTES:
        raise ValueError(
            "MPS arena size must be positive and below the addressing limit"
        )
    native = _native()
    # The new buffer has no producer work to synchronize. Consumption transfers
    # responsibility for its deleter to Torch's storage, not to this function.
    capsule = native.allocate(num_bytes)
    try:
        return torch.from_dlpack(capsule)
    except (RuntimeError, BufferError) as error:
        # Older Torch releases do not understand kDLMetal. Do not silently
        # return to the heap path that this allocator exists to avoid. Leave
        # unrelated conversion errors intact (including genuine OOM errors).
        if "device" in str(error).lower() and "support" in str(error).lower():
            raise RuntimeError(
                "This PyTorch build cannot import Metal DLPack storage. "
                "Use PyTorch 2.13 or newer for MPS rendering; Algan's CPU/CUDA "
                "dependency minimums are unchanged."
            ) from error
        raise
