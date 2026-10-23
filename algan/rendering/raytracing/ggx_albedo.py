"""Directional albedo of the path tracer's opaque GGX lobe.

The path tracer's specular lobe is ``D * G2 / (4 n.v)`` with the exact
height-correlated Smith ``G2`` and ``alpha = roughness**2``. Its Turquin
multiple-scattering compensation, ``1 + f0 (1 - E1) / E1``, is energy
preserving only when ``E1`` is the directional albedo of *that* lobe. It used
to come from Karis's split-sum fit (``_env_brdf_approx``), which describes a
different, darker lobe, so a white rough metal reflected up to 27% more than
arrived (roughness 0.25-0.75) and still too little at roughness 1.

This table is that lobe's albedo, split the usual way for Schlick Fresnel:
``E(f0) = f0 * A + B``, so ``E1 = A + B``. It rides at the end of the glass
energy table (``pt_energy_tables``), which the path tracer already copies into
the arena, so no kernel gains a parameter. ``scripts/generate_ggx_albedo.py``
regenerates it; nothing is integrated at import or render time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# Pure constants, so ``scripts/generate_ggx_albedo.py`` can read them without
# importing Algan. The row where this table starts inside
# ``pt_energy_tables()`` is ``GLASS_ENERGY_SHAPE[0]``.
GGX_ROUGHNESS_SIZE = 33
GGX_COSINE_SIZE = 33


@lru_cache(maxsize=1)
def ggx_albedo_table():
    """The shipped table, ``[roughness * cosine, 2]`` float32.

    The columns are ``A`` and ``B``. Axes are roughness and ``sqrt(n.v)``,
    both uniform on [0, 1].
    """
    path = Path(__file__).with_name("_ggx_albedo_lut.npz")
    with np.load(path, allow_pickle=False) as archive:
        table = archive["scale_bias"]
    shape = (GGX_ROUGHNESS_SIZE, GGX_COSINE_SIZE, 2)
    if table.shape != shape or table.dtype != np.float32:
        raise ValueError(f"Invalid GGX albedo table: {table.shape}, {table.dtype}")
    return table.reshape(-1, 2)


@lru_cache(maxsize=1)
def pt_energy_tables():
    """The glass energy table with the GGX albedo table appended.

    This is the one ``pt_glass_energy`` buffer the path tracer binds.
    """
    from algan.rendering.raytracing.glass_energy import glass_energy_table

    return np.concatenate((glass_energy_table(), ggx_albedo_table()), axis=0)
