"""The shipped blue-noise sampler tile, and its loader.

``data/blue_noise_tile_64.npy`` is a ``uint16`` ``64 x 64`` permutation of
``0 .. 4095``: per-pixel sampler keys, optimised so that the path tracer's
Monte Carlo error is distributed as blue noise in screen space rather than as
white noise (Heitz et al. 2019; ``DESIGN_path_tracer_roadmap.md`` section 7).
``scripts/generate_blue_noise_tile.py`` generates it and records the
parameters; this module only loads it.

The tile reaches the kernels as the tail of ``nee_meta`` (see
``path_tracer_taichi``'s ``_NM_BN_BASE``), so what the loader returns is a flat
**float32** CPU tensor in row-major order, ready to concatenate onto that
vector. Integer keys below ``2 ** 24`` are exact in float32, so nothing is
lost.

A missing or malformed file is not a render error: ``blue_noise_tile``
returns ``None`` after one warning and the caller renders with the switch off
(``path_tracer._build_nee_tables``). The alternative -- raising from inside a
render because a data file did not ship -- would trade a sampling refinement
for a broken install.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Edge of the tile, in pixels. Must match ``path_tracer_taichi.PT_BN_TILE``
#: (the kernel masks with ``PT_BN_TILE - 1``, so it also has to be a power of
#: two).
TILE_SIZE = 64

TILE_PATH = Path(__file__).parent / "data" / f"blue_noise_tile_{TILE_SIZE}.npy"


@lru_cache(maxsize=1)
def blue_noise_tile() -> torch.Tensor | None:
    """The tile as a flat ``[TILE_SIZE ** 2]`` float32 CPU tensor, or ``None``.

    Cached: the file is read, validated and converted once per process.
    """
    try:
        values = np.load(TILE_PATH)
    except OSError as exc:  # missing, truncated, unreadable
        logger.warning(
            "blue-noise sampler tile %s could not be read (%s); the path "
            "tracer will sample with the hashed per-pixel key instead",
            TILE_PATH,
            exc,
        )
        return None
    if values.shape != (TILE_SIZE, TILE_SIZE) or values.dtype != np.uint16:
        logger.warning(
            "blue-noise sampler tile %s is %s %s, expected uint16 (%d, %d); "
            "sampling with the hashed per-pixel key instead",
            TILE_PATH,
            values.dtype,
            values.shape,
            TILE_SIZE,
            TILE_SIZE,
        )
        return None
    flat = values.reshape(-1).astype(np.int64)
    if not np.array_equal(np.sort(flat), np.arange(flat.size)):
        # The construction guarantees a permutation, and the kernel's shift
        # relies on every pixel of the tile carrying a distinct key.
        logger.warning(
            "blue-noise sampler tile %s is not a permutation of 0..%d; "
            "sampling with the hashed per-pixel key instead",
            TILE_PATH,
            flat.size - 1,
        )
        return None
    return torch.from_numpy(flat.astype(np.float32))
