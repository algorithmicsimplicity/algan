"""CPU constants for the reciprocal two-sided rough-glass energy closure.

The shipped table contains *power* lost by ideal white single-scatter glass,
not a reflection-only GGX albedo and not radiance including eta squared.
Regenerate with ``scripts/generate_glass_energy_table.py``. Kernel lookups live
in ``glass_energy_taichi.py``; no persistent device fields survive a runtime
reset. The table is copied into the render's existing constant-data vector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

GLASS_ETA_SIZE = 65
GLASS_ROUGH_SIZE = 33
GLASS_MU_SIZE = 65
GLASS_TABLE_SHAPE = (GLASS_ETA_SIZE, GLASS_ROUGH_SIZE, 2, GLASS_MU_SIZE + 1)


@lru_cache(maxsize=1)
def glass_energy_table() -> torch.Tensor:
    """Load the immutable-by-convention, flat CPU float32 table once."""
    path = Path(__file__).with_name("data") / "glass_energy.npy"
    table = np.load(path, allow_pickle=False)
    if table.shape != GLASS_TABLE_SHAPE or table.dtype != np.float32:
        raise ValueError(f"Invalid rough-glass energy table layout: {path}")
    if not np.isfinite(table).all() or np.any((table < 0.0) | (table > 1.0)):
        raise ValueError(f"Invalid rough-glass energy table values: {path}")
    return torch.from_numpy(table.reshape(-1).copy())
