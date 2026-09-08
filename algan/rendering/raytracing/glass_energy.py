"""Immutable reciprocal rough-dielectric energy compensation table.

This is a coupled reflection/transmission *compensation*, not a microfacet
random walk. See DESIGN_rough_glass_energy.md for the transport equation,
colour/absorption bound, numerical error, and regeneration instructions.
No quadrature is performed at import or render time. The fixed-size table is copied
into the render arena once and is shared by every material and frame.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

GLASS_ROUGHNESS_SIZE = 17
GLASS_IOR_SIZE = 33
GLASS_COSINE_SIZE = 65
GLASS_ENERGY_SHAPE = (
    GLASS_ROUGHNESS_SIZE * GLASS_IOR_SIZE * (GLASS_COSINE_SIZE + 1),
    2,
)


@lru_cache(maxsize=1)
def glass_energy_table():
    """Decode the shipped table; columns are low- and high-IOR incidence.

    Each roughness/IOR block ends with its cosine-average missing power.

    Axes are roughness, sqrt(abs((eta-1)/(eta+1))), and sqrt(abs(cosine)).
    The caller owns the device/arena copy. A missing or damaged asset is an
    installation error, not a reason to silently render single-scatter glass.
    """
    path = Path(__file__).with_name("_glass_energy_lut.npz")
    with np.load(path, allow_pickle=False) as archive:
        packed = archive["loss_delta"]
    shape = (GLASS_ROUGHNESS_SIZE, GLASS_IOR_SIZE, GLASS_COSINE_SIZE, 2)
    if packed.shape != shape or packed.dtype != np.uint8:
        raise ValueError(f"Invalid glass energy table: {packed.shape}, {packed.dtype}")
    # Reconstruct the lossless modulo-256 differences along IOR and cosine.
    # This only shrinks the shipped asset; it adds no approximation.
    packed = packed.cumsum(axis=1, dtype=np.uint8).cumsum(axis=2, dtype=np.uint8)
    # Loss is bounded to [0,1]. Eight-bit storage introduces at most 0.5/255
    # absolute error; all device arithmetic and interpolation remain f32.
    values = packed.astype(np.float64) / 255.0
    # Exactly integrate the decoded piecewise-linear sqrt(mu) interpolant.
    # Recomputing this tiny weighted sum, rather than quantising its means
    # separately, preserves the shared budget and reciprocal normalisation.
    u = np.linspace(0.0, 1.0, GLASS_COSINE_SIZE)
    a, b = u[:-1], u[1:]
    fourth = b**4 - a**4
    fifth = 0.8 * (b**5 - a**5)
    left = (b * fourth - fifth) / (b - a)
    right = (fifth - a * fourth) / (b - a)
    means = np.sum(
        values[:, :, :-1, :] * left[None, None, :, None]
        + values[:, :, 1:, :] * right[None, None, :, None],
        axis=2,
        keepdims=True,
    )
    return np.concatenate((values, means), axis=2).astype(np.float32).reshape(
        GLASS_ENERGY_SHAPE
    )
