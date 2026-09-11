"""Typed, arena-owned metadata shared by deterministic shade kernels.

Integer addresses and counts never round-trip through float32. The two fixed
layouts are independent of render route, so every kernel sees initialized
fields rather than detecting optional payloads by an array's length.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import torch

LAYER_OFFSET = 0
ENV_INTENSITY = 1
FAR_CLIP = 2
FLOAT_WORDS = 3

ENV_OFFSET = 0
ENV_WIDTH = 1
ENV_HEIGHT = 2
MAX_BOUNCES = 3
GLOSS_BASE = 4
INT_WORDS = 5


class RenderMetadata(NamedTuple):
    floats: torch.Tensor
    ints: torch.Tensor


def allocate_render_metadata(
    memory, layer_offset, *, env_meta=None, far_clip=0.0, max_bounces=0
):
    """Allocate initialized float/int tables, validating before taking arena bytes.

    The optional environment tuple is (texel offset, width, height, intensity).
    Addresses/counts must be nonnegative signed-int32 integers. Validation is
    host-only; no device reduction or float-to-integer conversion is involved.
    The caller's batch scope owns both tables. GLOSS_BASE starts at zero and
    is rewritten with an integer accumulator offset for each glossy tile.
    """
    offset, width, height, intensity = env_meta or (0, 0, 0, 0.0)
    values = []
    for name, value in zip(
        (
            "environment offset",
            "environment width",
            "environment height",
            "max_bounces",
        ),
        (offset, width, height, max_bounces),
    ):
        value = operator.index(value)
        if not 0 <= value < 2**31:
            raise ValueError(f"{name} must fit nonnegative int32, got {value}")
        values.append(value)
    float_source = torch.tensor(
        [float(layer_offset), float(intensity), float(far_clip)],
        dtype=torch.float32,
        device="cpu",
    )
    int_source = torch.tensor([*values, 0], dtype=torch.int32, device="cpu")
    floats = memory.get_tensor((FLOAT_WORDS,), torch.float32)
    ints = memory.get_tensor((INT_WORDS,), torch.int32)
    floats.copy_(float_source)
    ints.copy_(int_source)
    return RenderMetadata(floats, ints)
