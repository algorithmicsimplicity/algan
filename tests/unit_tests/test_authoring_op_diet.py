"""Authoring helpers rewritten for fewer ops build exactly what they used to.

Each rewrite here replaced a per-element or per-slot loop of tiny tensor ops
with one op, on the promise that the result is the same tensor -- values to
the bit, and for ``cast_to_tensor`` shape, strides, dtype, type and storage
too. These tests keep the loops they replaced as references and hold the
rewrites to them on the inputs most likely to tell them apart: signed zeros,
denormals, out-of-range numbers, subclasses, views and odd dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan.animatable_base.mob_layout import MobLayoutMixin
from algan.animatable_base.mob_morph import _gather_padded_objects
from algan.constants.color import BLUE
from algan.utils import tensor_utils
from algan.utils.tensor_utils import cast_to_tensor


def _reference_cast(x):
    """``cast_to_tensor`` before its fast paths, verbatim."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    if isinstance(x, torch.Tensor):
        while x.dim() < 3:
            x = x.unsqueeze(0)
        return x
    if isinstance(x, (list, tuple)):
        x = tensor_utils._cast_to_tensor_recursive(x).squeeze(-1)
        return x.view(1, -1, x.shape[-1])
    if x is None:
        x = 0
    try:
        return torch.tensor((x,), dtype=torch.get_default_dtype()).view(1, 1, 1)
    except (TypeError, ValueError, RuntimeError):
        return x


_F32_MAX = torch.finfo(torch.float32).max

CAST_INPUTS = [
    0.5,
    -0.0,
    1e-46,  # rounds to a signed zero in float32
    -1e-46,
    5e-324,
    _F32_MAX,
    _F32_MAX * (1 + 2**-30),  # beyond float32's range: an infinity, not an error
    1e39,
    -1e39,
    float("nan"),
    float("inf"),
    1,
    0,
    True,
    None,
    2**60 + 1,
    "a string",
    (0.0, 1.0, 2.0),
    (1, 2, 3),
    [0.1, 7],
    (0.5,),
    (True, 0.5),
    (16777217, 0.5),  # an int float32 cannot hold exactly
    (1e39, 0.5),
    ((1.0, 2.0), (3.0, 4.0)),
    (BLUE, BLUE),
    (torch.tensor([1.0, 2.0, 3.0]), 0.5),
    (),
    torch.tensor(2.0),
    torch.arange(3.0),
    torch.arange(12.0).view(4, 3),
    torch.arange(12.0).view(3, 4).t(),  # a non-contiguous view
    torch.arange(6.0).view(1, 2, 3),
    BLUE,
    BLUE.unsqueeze(0),
    np.float64(0.25),
    np.array([1.0, 2.0]),
    torch.tensor([1, 2, 3]),
]


def _describe(value):
    if not torch.is_tensor(value):
        return value
    bits = value.contiguous()
    if bits.is_floating_point():
        bits = bits.view(torch.uint8)
    return (
        type(value),
        tuple(value.shape),
        value.stride(),
        value.dtype,
        bits.flatten().tolist(),
    )


def _cast_outcome(cast, x):
    try:
        return _describe(cast(x))
    except Exception as error:  # the type of failure must match too
        return type(error)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", CAST_INPUTS, ids=lambda v: repr(v)[:30])
def test_cast_to_tensor_builds_the_tensor_it_always_built(value, dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        assert _cast_outcome(cast_to_tensor, value) == _cast_outcome(
            _reference_cast, value
        )
        if torch.is_tensor(value):
            # Tensors are reshaped views of the caller's storage, as before.
            assert cast_to_tensor(value).data_ptr() == value.data_ptr()
    finally:
        torch.set_default_dtype(previous)


def _reference_box_corners(lower_corner, upper_corner):
    out = torch.empty(*lower_corner.shape[:-2], 8, 3, device=lower_corner.device)
    lower_corner = lower_corner.squeeze(-2)
    upper_corner = upper_corner.squeeze(-2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = torch.tensor((i, j, k), device=lower_corner.device)
                out[..., i * 4 + j * 2 + k, :] = (
                    lower_corner * (1 - a) + (a) * upper_corner
                )
    return out


@pytest.mark.parametrize("shape", [(1, 1, 3), (4, 1, 3), (1, 3), (2, 5, 1, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_box_corners_match_the_per_corner_loop_bit_for_bit(shape, dtype):
    generator = torch.Generator().manual_seed(7)
    for _ in range(25):
        lower = torch.randn(shape, generator=generator, dtype=dtype)
        upper = lower + torch.rand(shape, generator=generator, dtype=dtype)
        # Signed zeros on both sides: 0 * x is -0.0 for a negative x, which is
        # what decides the sign of a zero corner.
        lower[torch.rand(shape, generator=generator) < 0.3] = -0.0
        lower[torch.rand(shape, generator=generator) < 0.2] = 0.0
        upper[torch.rand(shape, generator=generator) < 0.3] = 0.0
        upper[torch.rand(shape, generator=generator) < 0.2] = -0.0
        assert _describe(MobLayoutMixin._box_corners(lower, upper)) == _describe(
            _reference_box_corners(lower, upper)
        )


def _reference_padding(value, sources):
    seen = set()
    aligned = []
    for source in sources:
        segment = value.select(-3, source)
        if source in seen:
            segment = segment[..., -1:, :].expand_as(segment).clone()
        else:
            seen.add(source)
        aligned.append(segment)
    return torch.stack(aligned, dim=-3)


@pytest.mark.parametrize("shape", [(5, 4, 3), (2, 5, 4, 3), (3, 1, 7), (6, 0, 3)])
@pytest.mark.parametrize("grow", [1, 2, 7])
def test_padding_an_object_batch_matches_the_per_slot_loop(shape, grow):
    value = torch.randn(shape, generator=torch.Generator().manual_seed(3))
    current = value.shape[-3]
    target = current + grow
    sources = [(slot * current) // target for slot in range(target)]
    assert _describe(_gather_padded_objects(value, sources)) == _describe(
        _reference_padding(value, sources)
    )


def test_padding_a_non_contiguous_batch_matches_the_per_slot_loop():
    value = torch.randn(4, 3, 5).transpose(0, 1)
    sources = [0, 0, 1, 1, 2]
    assert _describe(_gather_padded_objects(value, sources)) == _describe(
        _reference_padding(value, sources)
    )
