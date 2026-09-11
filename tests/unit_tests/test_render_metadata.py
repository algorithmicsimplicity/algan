"""Integer render addresses do not round-trip through float32 metadata."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.render_metadata import (
    ENV_HEIGHT,
    ENV_INTENSITY,
    ENV_OFFSET,
    ENV_WIDTH,
    FAR_CLIP,
    FLOAT_WORDS,
    GLOSS_BASE,
    INT_WORDS,
    LAYER_OFFSET,
    MAX_BOUNCES,
    allocate_render_metadata,
)
from algan.utils.memory_utils import ManualMemory


def _memory():
    return ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=4096)


@pytest.mark.parametrize(
    "offset", [0, 2**23 + 1, 2**24 - 1, 2**24 + 1, 2**25 + 3, 2**31 - 1]
)
def test_render_metadata_keeps_integer_offsets_exact(offset):
    memory = _memory()
    memory._poison = 255
    data = allocate_render_metadata(
        memory,
        11.0,
        env_meta=(offset, 31, 17, 1.5),
        far_clip=19.0,
        max_bounces=9,
    )
    assert data.floats.shape == (FLOAT_WORDS,)
    assert data.ints.shape == (INT_WORDS,)
    assert data.floats.dtype == torch.float32
    assert data.ints.dtype == torch.int32
    assert data.ints[ENV_OFFSET].item() == offset
    assert data.ints[ENV_WIDTH].item() == 31
    assert data.ints[ENV_HEIGHT].item() == 17
    assert data.ints[MAX_BOUNCES].item() == 9
    assert data.ints[GLOSS_BASE].item() == 0
    data.ints[GLOSS_BASE] = offset
    assert data.ints[GLOSS_BASE].item() == offset
    assert data.floats[LAYER_OFFSET].item() == 11
    assert data.floats[ENV_INTENSITY].item() == 1.5
    assert data.floats[FAR_CLIP].item() == 19
    for tensor in data:
        assert tensor.untyped_storage()._cdata == memory.data.untyped_storage()._cdata


def test_render_metadata_without_environment_initializes_every_word():
    memory = _memory()
    memory._poison = 255
    data = allocate_render_metadata(memory, 7)
    assert data.floats.cpu().tolist() == [7.0, 0.0, 0.0]
    assert data.ints.cpu().tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("value", "exception"), [(-1, ValueError), (2**31, ValueError), (1.25, TypeError)]
)
@pytest.mark.parametrize("field", ["offset", "width", "height", "bounces"])
def test_invalid_render_integer_is_rejected_before_arena_allocation(
    value, exception, field
):
    memory = _memory()
    values = {"offset": 0, "width": 1, "height": 1, "bounces": 2, field: value}
    before = memory.get_pointers()
    with pytest.raises(exception):
        allocate_render_metadata(
            memory,
            0,
            env_meta=(values["offset"], values["width"], values["height"], 1.0),
            max_bounces=values["bounces"],
        )
    assert memory.get_pointers() == before
