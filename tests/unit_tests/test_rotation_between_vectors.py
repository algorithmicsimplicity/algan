"""``get_rotation_between_3d_vectors`` near 0 and 180 degrees.

It took the angle as ``arccos`` of the cosine similarity. For (anti)parallel
vectors the cosine rounds to just outside [-1, 1] often enough that the angle
came back NaN, so ``look()`` and ``look_at()`` raised on about 30% of
directions when the Mob already faced the target. The angle is now
``atan2(|cross|, dot)``, which is finite everywhere.
"""

from __future__ import annotations

import pytest
import torch

from algan import Scene, Square
from algan.geometry.geometry import (
    get_rotation_around_axis,
    get_rotation_between_3d_vectors,
)

N = 20_000


def _pairs(kind: str):
    g = torch.Generator().manual_seed(1234)
    v = torch.randn(N, 3, generator=g)
    w = torch.randn(N, 3, generator=g)
    k = torch.rand(N, 1, generator=g) * 9 + 0.1
    if kind == "parallel":
        return v, k * v
    if kind == "antiparallel":
        return v, -k * v
    if kind == "near":
        return v, v + 1e-4 * w
    return v, w


@pytest.mark.parametrize("kind", ["parallel", "antiparallel", "near", "generic"])
def test_rotation_between_vectors_is_valid_and_aligns(kind):
    v1, v2 = _pairs(kind)
    deg, axis = get_rotation_between_3d_vectors(v1, v2, dim=-1)
    assert deg.shape == (N, 1)
    assert torch.isfinite(deg).all()
    assert torch.isfinite(axis).all()
    assert (deg >= 0).all()
    assert (deg <= 180 + 1e-3).all()
    assert torch.allclose(axis.norm(dim=-1), torch.ones(N), atol=1e-4)
    R = get_rotation_around_axis(deg, axis, dim=-1)
    assert torch.isfinite(R).all()
    assert (torch.linalg.det(R) > 0).all()
    out = (torch.nn.functional.normalize(v1, dim=-1).unsqueeze(1) @ R).squeeze(1)
    target = torch.nn.functional.normalize(v2, dim=-1)
    assert (out - target).abs().max() < 1e-3


def test_look_twice_does_not_raise():
    g = torch.Generator().manual_seed(7)
    dirs = torch.randn(100, 3, generator=g)
    for d in dirs:
        with Scene():
            m = Square()
            m.look(d)
            m.look(d)
        with Scene():
            m = Square()
            m.look(d)
            m.look(-d)
