from __future__ import annotations

import numpy as np
import pytest
import torch

from algan.mobs.three_d_models.mesh import _face_component_ids
from algan.rendering.raytracing.raster_pipeline import (
    _classify_exact_fragments,
    reset_exact_aa_fallback_counts,
)
from algan.rendering.raytracing.raster_taichi import (
    AA_FALLBACK_DEPTH_UNCERTAINTY,
    AA_FALLBACK_MULTIPLE_PARTIAL_GROUPS,
    AA_FALLBACK_SELF_OVERLAP,
)
from algan.settings._startup import _RENDER_DEVICE

_DEVICE = torch.device(_RENDER_DEVICE)


def _world_from_screen(screen_triangles, depths):
    """World points projecting to ``screen_triangles`` for the test camera."""
    screen = np.asarray(screen_triangles, dtype=np.float32)
    z = np.broadcast_to(np.asarray(depths, dtype=np.float32), screen.shape[:-1])
    return np.stack(
        ((2.0 * screen[..., 0] - 1.0) * z, (2.0 * screen[..., 1] - 1.0) * z, z), axis=-1
    )


def _classify(
    screen_triangles,
    depths,
    components,
    coverages,
    opaque=None,
    transport=False,
    stub_extra=False,
):
    screen_triangles = np.asarray(screen_triangles, dtype=np.float32)
    count = len(screen_triangles)
    if opaque is None:
        opaque = np.ones(count, dtype=bool)
    tri_screen = torch.zeros((1, count, 13), dtype=torch.float32, device=_DEVICE)
    tri_screen[0, :, 0:3] = torch.as_tensor(screen_triangles[..., 0], device=_DEVICE)
    tri_screen[0, :, 3:6] = torch.as_tensor(screen_triangles[..., 1], device=_DEVICE)
    tri_screen[0, :, 6:9] = 1.0
    tri_screen[0, :, 9] = 1.0

    world = _world_from_screen(screen_triangles, depths).reshape(1, count, 9)
    merged = {
        "num_triangles": count,
        "num_circuits": 0,
        "tri_component": torch.as_tensor(
            components, dtype=torch.int32, device=_DEVICE
        ).view(1, -1),
        "tri_pos": torch.as_tensor(world, dtype=torch.float32, device=_DEVICE),
        "circuit_meta": torch.zeros((1, 1, 25), device=_DEVICE),
    }
    if transport is not False:
        extra = torch.zeros((1, count, 12), dtype=torch.float32, device=_DEVICE)
        extra[..., 0:6:2] = -1.0
        transport_mask = np.asarray(transport, dtype=bool)
        if transport_mask.ndim == 0:
            transport_mask = np.full(count, bool(transport_mask))
        extra[0, torch.as_tensor(transport_mask, device=_DEVICE), 0:6:2] = 1.0
        merged["tri_extra"] = extra
    elif stub_extra:
        merged["tri_extra"] = torch.zeros(
            (1, 1, 12), dtype=torch.float32, device=_DEVICE
        )
    # All fragments belong to local pixel zero.  The low key bits only provide
    # the already-sorted representative depth; interval certification below is
    # computed independently from the actual planes.
    depth_key = torch.arange(1, count + 1, dtype=torch.float32, device=_DEVICE)
    key = depth_key.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ref = torch.arange(count, dtype=torch.int32, device=_DEVICE)
    cov = torch.as_tensor(coverages, dtype=torch.float32, device=_DEVICE)
    opaque_t = torch.as_tensor(opaque, dtype=torch.bool, device=_DEVICE)
    covered = torch.zeros((1,), dtype=torch.int64, device=_DEVICE)
    counts = torch.tensor((count,), dtype=torch.int64, device=_DEVICE)
    camera = torch.tensor(((0.0, 0.0, 0.0),), device=_DEVICE)
    screen_point = torch.tensor(((0.0, 0.0, 1.0),), device=_DEVICE)
    basis_x = torch.tensor(((1.0, 0.0, 0.0),), device=_DEVICE)
    basis_y = torch.tensor(((0.0, 1.0, 0.0),), device=_DEVICE)
    reset_exact_aa_fallback_counts()
    _group, resolved, reasons, _offsets = _classify_exact_fragments(
        key,
        ref,
        cov,
        opaque_t,
        covered,
        counts,
        merged,
        tri_screen,
        camera,
        screen_point,
        basis_x,
        basis_y,
        0,
        1,
        1,
        0.5,
        0.5,
    )
    return resolved.detach().cpu(), int(reasons.item())


_PARTITION = np.array(
    [
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ],
    dtype=np.float32,
)


def test_connected_same_facing_partition_is_scalar_resolvable():
    resolved, reason = _classify(
        _PARTITION,
        np.ones((2, 3), dtype=np.float32),
        components=(0, 0),
        coverages=(0.5, 0.5),
    )

    assert reason == 0
    # Conditional coverage makes ordinary alpha compositing add 0.5 + 0.5.
    torch.testing.assert_close(resolved, torch.tensor((0.5, 1.0)))


def test_material_stub_is_not_indexed_as_owner_wide():
    resolved, reason = _classify(
        _PARTITION,
        np.ones((2, 3), dtype=np.float32),
        components=(0, 0),
        coverages=(0.5, 0.5),
        stub_extra=True,
    )

    assert reason == 0
    torch.testing.assert_close(resolved, torch.tensor((0.5, 1.0)))


def test_connected_multi_cell_partition_is_scalar_resolvable():
    triangles = []
    for y in range(2):
        for x in range(2):
            x0, x1 = 0.5 * x, 0.5 * (x + 1)
            y0, y1 = 0.5 * y, 0.5 * (y + 1)
            triangles.extend(
                (
                    ((x0, y0), (x1, y0), (x1, y1)),
                    ((x0, y0), (x1, y1), (x0, y1)),
                )
            )
    triangles = np.asarray(triangles, dtype=np.float32)
    resolved, reason = _classify(
        triangles,
        np.ones((len(triangles), 3), dtype=np.float32),
        components=np.zeros(len(triangles), dtype=np.int32),
        coverages=np.full(len(triangles), 0.125, dtype=np.float32),
    )

    assert reason == 0
    assert abs(float(1.0 - torch.prod(1.0 - resolved)) - 1.0) <= 1e-6


@pytest.mark.parametrize(
    "triangles",
    [
        _PARTITION,  # disjoint regions with identical scalar areas
        np.repeat(_PARTITION[:1], 2, axis=0),  # coincident regions
        np.array(
            [
                [[-0.2, 0.1], [1.2, 0.1], [0.5, 0.9]],
                [[0.1, -0.2], [0.9, 0.5], [0.1, 1.2]],
            ],
            dtype=np.float32,
        ),  # crossing regions
    ],
)
def test_independent_partial_regions_are_not_scalar_resolvable(triangles):
    _resolved, reason = _classify(
        triangles,
        np.ones((2, 3), dtype=np.float32),
        components=(0, 1),
        coverages=(0.5, 0.5),
    )

    assert reason & AA_FALLBACK_MULTIPLE_PARTIAL_GROUPS


def test_same_component_overlap_is_rejected_even_with_one_full_fragment():
    triangles = np.repeat(_PARTITION[:1], 2, axis=0)
    _resolved, reason = _classify(
        triangles,
        np.ones((2, 3), dtype=np.float32),
        components=(0, 0),
        coverages=(1.0, 0.5),
    )

    assert reason & AA_FALLBACK_SELF_OVERLAP


def test_strict_full_layer_order_is_scalar_but_crossing_depth_is_not():
    full = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    triangles = np.repeat(full, 2, axis=0)

    _resolved, ordered_reason = _classify(
        triangles,
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        components=(0, 1),
        coverages=(1.0, 1.0),
    )
    _resolved, crossing_reason = _classify(
        triangles,
        np.array([[1.0, 10.0, 10.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        components=(0, 1),
        coverages=(1.0, 1.0),
    )

    assert ordered_reason == 0
    assert crossing_reason & AA_FALLBACK_DEPTH_UNCERTAINTY


def test_one_partial_region_between_ordered_full_layers_is_scalar_resolvable():
    full = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    triangles = np.concatenate((full, _PARTITION[:1], full), axis=0)

    _resolved, reason = _classify(
        triangles,
        np.array(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            dtype=np.float32,
        ),
        components=(0, 1, 2),
        coverages=(1.0, 0.5, 1.0),
        opaque=(False, False, False),
    )

    assert reason == 0


def test_partial_secondary_transport_requires_spatial_fallback():
    _resolved, partial_reason = _classify(
        _PARTITION[:1],
        np.ones((1, 3), dtype=np.float32),
        components=(0,),
        coverages=(0.5,),
        transport=True,
    )
    _resolved, full_reason = _classify(
        _PARTITION[:1],
        np.ones((1, 3), dtype=np.float32),
        components=(0,),
        coverages=(1.0,),
        transport=True,
    )

    assert partial_reason & AA_FALLBACK_DEPTH_UNCERTAINTY
    assert full_reason == 0


def test_partial_layer_over_full_transport_requires_spatial_fallback():
    full = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    triangles = np.concatenate((_PARTITION[:1], full), axis=0)
    _resolved, reason = _classify(
        triangles,
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        components=(0, 1),
        coverages=(0.5, 1.0),
        transport=(False, True),
    )

    assert reason & AA_FALLBACK_DEPTH_UNCERTAINTY


def test_face_components_require_shared_edges_not_only_vertices():
    faces = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 3],  # shares edge (1, 2) with face zero
            [2, 4, 5],  # touches the first component at vertex 2 only
        ],
        dtype=torch.int64,
    )

    ids = _face_component_ids(faces)

    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
