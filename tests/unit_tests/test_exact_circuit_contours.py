from __future__ import annotations

import math

import numpy as np
import torch

from algan.rendering.raytracing.exact_coverage import (
    EXACT_REASON_SELF_OVERLAP,
    _build_one_circuit,
    build_exact_circuit_contours,
)


def _rows(points, *, visible=True):
    points = np.asarray(points, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    return np.column_stack(
        (
            points,
            following,
            np.full((len(points),), float(visible), dtype=np.float64),
        )
    )


def _boundary_area(edges):
    return 0.5 * float(np.sum(edges[:, 0] * edges[:, 3] - edges[:, 1] * edges[:, 2]))


def test_filled_contour_builds_outward_total_and_inward_fill_regions():
    square = _rows(((0, 0), (2, 0), (2, 2), (0, 2)))

    total, fill, reason = _build_one_circuit(
        square,
        filled=True,
        border_width=0.25,
        outline_width=0.3,
        chord_tolerance=0.01,
    )

    assert reason == 0
    assert _boundary_area(total) > 4.0
    assert 0.0 < _boundary_area(fill) < 4.0
    # Outward convex corners are explicit round arcs, not a slope-independent
    # SDF fade or an implicit sample pattern.
    assert len(total) > 4


def test_open_stroke_has_round_caps_and_conserves_capsule_area():
    # The final invisible edge is the synthetic fill closure.  It delimits an
    # open visible subpath and must become two round caps, not a rendered edge.
    line = np.array(
        [[0.0, 0.0, 3.0, 0.0, 1.0], [3.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    width = 0.8

    total, fill, reason = _build_one_circuit(
        line,
        filled=False,
        border_width=width,
        outline_width=0.3,
        chord_tolerance=0.002,
    )

    assert reason == 0
    assert fill.shape == (0, 4)
    expected = 3.0 * width + math.pi * (0.5 * width) ** 2
    assert abs(_boundary_area(total) - expected) <= 0.01
    assert len(total) > 6


def test_crossing_fill_is_tagged_for_ray_fallback():
    bow_tie = _rows(((0, 0), (2, 2), (0, 2), (2, 0)))

    _total, _fill, reason = _build_one_circuit(
        bow_tie,
        filled=True,
        border_width=0.0,
        outline_width=0.3,
        chord_tolerance=0.01,
    )

    assert reason == EXACT_REASON_SELF_OVERLAP


def test_projected_contours_follow_camera_raw_dot_product_contract():
    # Camera screen rows are not generally orthogonal after rotation and
    # non-uniform screen scaling. Projection uses row 2 as the plane normal,
    # then raw dot products against rows 0/1 (Camera.get_render_screen_basis).
    points = np.array(
        [[-0.5, -0.4], [0.6, -0.4], [0.6, 0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    edge_rows = _rows(points)
    edges = torch.as_tensor(edge_rows, dtype=torch.float32).unsqueeze(0)
    offsets = torch.tensor((0, 4), dtype=torch.int32)
    edge_circuit = torch.zeros((4,), dtype=torch.int32)

    meta = torch.zeros((1, 1, 25), dtype=torch.float32)
    meta[0, 0, 0:3] = torch.tensor((0.1, -0.2, 3.0))
    meta[0, 0, 3:6] = torch.tensor((0.0, 0.0, 1.0))
    meta[0, 0, 6:9] = torch.tensor((1.0, 0.0, 0.0))
    meta[0, 0, 9:12] = torch.tensor((0.0, 1.0, 0.0))
    meta[0, 0, 13] = 1.0

    camera = torch.tensor(((0.0, 0.0, 0.0),), dtype=torch.float32)
    screen_point = torch.tensor(((0.0, 0.0, 1.0),), dtype=torch.float32)
    screen_basis = torch.tensor(
        (((0.55, 0.0, 0.18), (0.08, 0.42, 0.06), (0.2, 0.1, 1.0)),),
        dtype=torch.float32,
    )
    result = build_exact_circuit_contours(
        edges,
        offsets,
        edge_circuit,
        meta,
        camera,
        screen_point,
        screen_basis,
        160,
        90,
        0.0,
        0.01,
    )

    local = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
    world = local + np.array((0.1, -0.2, 3.0), dtype=np.float32)
    basis = screen_basis[0].numpy()
    normal = basis[2]
    amount = float(np.dot(screen_point[0].numpy(), normal)) / (world @ normal)
    hit = amount[:, None] * world
    relative = hit - screen_point[0].numpy()
    expected = np.column_stack(
        (
            (relative @ basis[0]) * 45.0 + 80.0,
            (relative @ basis[1]) * 45.0 + 45.0,
        )
    )
    actual = result.total_edges[0, :, :2].cpu().numpy()
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    actual = actual[np.lexsort((actual[:, 1], actual[:, 0]))]

    assert int(result.reasons[0, 0]) == 0
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
