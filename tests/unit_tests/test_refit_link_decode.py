"""A runtime child index must preserve packed BVH links, including NaN bits."""

import pytest
import torch

from algan.rendering.raytracing.raytrace_kernels_taichi import NODE_ARG, _refit_link
from algan.rendering.raytracing.stbvh import (
    bvh_arity,
    bvh_block_f16,
    bvh_node_words,
    kernel_blocks,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
from algan.taichi_compat import ti


@ti.kernel
def _decode_links(
    blocks: NODE_ARG, choices: ti.types.ndarray(), out: ti.types.ndarray()
):
    for i in range(choices.shape[0]):
        out[i] = _refit_link(i // bvh_arity, choices[i], blocks)


def _kernel_view(blocks, fmt):
    """The block array as the kernels take it: ``native`` is
    ``stbvh.kernel_blocks`` (words, or the lane vectors under
    ``ALGAN_BVH_NODE_WORDS=0``); the others force one word format.
    """
    if fmt == "native":
        return kernel_blocks(blocks)
    lanes = blocks.reshape(blocks.shape[0], -1)
    if fmt == "scalar":
        return lanes
    return lanes.view(torch.int64 if fmt == "i64" else torch.int32)


@pytest.mark.parametrize("fmt", ["native", "i64", "i32", "scalar"])
def test_refit_links_preserve_all_bits_at_dynamic_child_indices(fmt):
    if fmt != "native" and not bvh_node_words:
        pytest.skip("ALGAN_BVH_NODE_WORDS=0 takes only the lane vectors")
    init_taichi()
    device = render_device()
    # Internal, invalid, leaf, opaque and non-casting link words. Several
    # halves represent NaNs as f16: numeric casts would corrupt their payload.
    words = (
        torch.tensor(
            [0, 12345, -1, -2147483648, -1073741823, -1610612731, 65535, 2147483647],
            dtype=torch.int32,
        )
        .repeat(bvh_arity)
        .reshape(8, bvh_arity)
    )
    dtype = torch.float16 if bvh_block_f16 else torch.float32
    blocks = torch.zeros((8, 8, bvh_arity), dtype=dtype)
    if bvh_block_f16:
        blocks.view(torch.int16)[:, 6] = (words & 65535).to(torch.int16)
        blocks.view(torch.int16)[:, 7] = (words >> 16).to(torch.int16)
    else:
        blocks.view(torch.int32)[:, 6] = words
    choices = torch.arange(bvh_arity - 1, -1, -1, dtype=torch.int32).repeat(8)
    out = torch.empty(choices.numel(), dtype=torch.int32, device=device)
    _decode_links(_kernel_view(blocks.to(device), fmt), choices.to(device), out)
    assert torch.equal(out.cpu(), words.flip(1).reshape(-1))
