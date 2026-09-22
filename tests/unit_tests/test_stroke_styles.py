"""Stroke geometry, compatibility, and timeline/render integration."""

from __future__ import annotations

import contextlib

import numpy as np
import pathops
import pytest
import torch

import algan as a
from algan.rendering.stroke_outline import _path, _regions

pytestmark = pytest.mark.usefixtures("fresh_scene")


def _polyline(points):
    points = np.asarray(points, dtype=float)
    return np.array(
        [
            [p, p + (q - p) / 3, p + 2 * (q - p) / 3, q]
            for p, q in zip(points, points[1:])
        ]
    )


def _outline(points, cap="butt", join="round", limit=4, width=0.4):
    curves, _ = _regions(
        _polyline(points), width, (cap, join, limit), False, False, 1e-5
    )[0]
    return _path(curves)


@pytest.mark.parametrize(
    ("cap", "area", "extent"),
    [("butt", 0.8, 1), ("square", 0.96, 1.2), ("round", 0.8 + np.pi * 0.04, 1.2)],
)
def test_caps_match_their_geometric_definition(cap, area, extent):
    outline = _outline([[-1, 0], [1, 0]], cap=cap)
    assert abs(outline.area) == pytest.approx(area, abs=1e-4)
    np.testing.assert_allclose(outline.bounds, (-extent, -0.2, extent, 0.2), atol=1e-6)


def test_join_geometry_and_miter_limit():
    points = [[-1, 0], [0, 0], [0, 1]]
    areas = [
        abs(_outline(points, join=join).area) for join in ("bevel", "round", "miter")
    ]
    assert areas[0] < areas[1] < areas[2]
    assert abs(_outline(points, join="miter", limit=1).area) == pytest.approx(
        areas[0], abs=1e-5
    )


def test_closed_paths_have_joins_but_no_caps():
    points = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
    outlines = [
        _outline(points, cap=cap, join="miter") for cap in ("butt", "round", "square")
    ]
    assert all(list(p) == list(outlines[0]) for p in outlines)
    assert abs(outlines[0].area) == pytest.approx(3.2, abs=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap_style": "triangle"},
        {"joint_type": "square"},
        {"miter_limit": 0},
        {"miter_limit": float("inf")},
    ],
)
def test_invalid_styles_raise(kwargs):
    with pytest.raises(a.AlganConfigurationError):
        a.Line(**kwargs)


@pytest.mark.fast
def test_style_survives_clone_and_compatible_packing():
    from algan.utils.mob_utils import batch_mobs

    line = a.Line(cap_style="square", joint_type="miter", miter_limit=2)
    clone = line.clone()
    packed = batch_mobs([line, clone])
    assert (
        clone._stroke_style_key()
        == packed._stroke_style_key()
        == ("square", "miter", 2)
    )
    assert packed.get_render_primitives().stroke_style == ("square", "miter", 2)
    with pytest.raises(a.AlganConfigurationError, match="different stroke styles"):
        batch_mobs([line, a.Line(cap_style="butt")])


def test_manim_styles_round_trip_and_delegated_cap_edit():
    import algan.manim as mn

    line = mn.Line(
        cap_style=mn.CapStyleType.SQUARE,
        joint_type=mn.LineJointType.MITER,
        miter_limit=2,
    )
    assert line._stroke_style_key() == ("square", "miter", 2)
    assert line.get_manim_mobject().cap_style == mn.CapStyleType.SQUARE
    with a.Off():
        line.set_cap_style(mn.CapStyleType.BUTT)
    assert line._stroke_style_key() == ("butt", "miter", 2)
    line.cap_style = mn.CapStyleType.SQUARE
    assert line.get_manim_mobject().cap_style == mn.CapStyleType.SQUARE


def test_background_strokes_are_diagnosed():
    with pytest.warns(a.UnsupportedFeatureWarning, match="Background strokes"):
        a.Line(background_stroke_width=5)
    import algan.manim as mn

    with pytest.warns(a.UnsupportedFeatureWarning, match="Background strokes"):
        mn.Line(background_stroke_width=5)


def _scene():
    return a.Scene(
        a.PREVIEW.set(resolution=(128, 96), frames_per_second=4), background=a.BLACK
    )


def _frames(scene, indices=(0,)):
    with contextlib.closing(
        scene.get_frames(0, len(indices), frame_indices=indices, post_processes=())
    ) as stream:
        return torch.cat(list(stream))


def test_caps_reach_the_renderer_and_keep_width():
    scene = _scene()
    with a.Off():
        line = a.Line(start=a.LEFT, end=a.RIGHT, stroke_width=60, color=a.WHITE).spawn()
    frames = []
    for cap in ("butt", "round", "square"):
        line.cap_style = cap
        frames.append(_frames(scene)[0, ..., 0])
    masks = [frame > 128 for frame in frames]
    counts = [int(mask.sum()) for mask in masks]
    assert counts[0] < counts[1] < counts[2]
    center_heights = [int(mask[:, 64].sum()) for mask in masks]
    assert max(center_heights) - min(center_heights) <= 2


def test_animated_width_and_camera_views_preserve_sparse_seeking():
    scene = _scene()
    with a.Off():
        line = a.Line(stroke_width=15, cap_style="square", color=a.RED).spawn()
        view = a.CameraView(resolution=(40, 30), height=2).focus_on(line, 0.8)
        view.move_to(a.RIGHT * 3 + a.OUT).spawn()
    with a.Sync():
        line.stroke_width = 60
        line.move(a.LEFT)
        view.camera.move(a.LEFT)
    frames = _frames(scene, (0, 2, 4))
    assert not torch.equal(frames[0], frames[-1])
    again = _frames(scene, (2,))
    assert (frames[1].int() - again[0].int()).abs().max() <= 2
    assert line.cap_style == "square"
    assert line.stroke_width.max() == 60


def test_filled_border_and_holes_keep_their_regions():
    outer = _polyline([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    inner = _polyline(
        [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4], [-0.4, -0.4]]
    )
    curves = np.concatenate((outer, inner))
    fill, border = _regions(curves, 0.1, ("square", "miter", 4), True, True, 1e-5)
    fill, border = _path(fill[0]), _path(border[0])
    assert abs(fill.area) + abs(border.area) == pytest.approx(4 - 0.64, abs=1e-5)
    overlap = pathops.op(fill, border, pathops.PathOp.INTERSECTION)
    assert abs(overlap.area) < 1e-6


def test_translucent_fill_and_border_do_not_double_opacity():
    scene = _scene()
    with a.Off():
        square = a.Square(
            color=a.WHITE,
            stroke_color=a.WHITE,
            opacity=0.5,
            joint_type="miter",
            stroke_width=45,
        ).spawn()
    outlined = _frames(scene)[0]
    with a.Off():
        square.stroke_width = 0
    plain = _frames(scene)[0]
    assert outlined.max().item() <= plain.max().item() + 2
    # A shared antialiased edge can redistribute subpixel coverage when a
    # region is split, but must preserve the total light and never composite
    # the fill and border as two full-opacity layers.
    assert outlined.float().sum().item() == pytest.approx(
        plain.float().sum().item(), rel=0.01
    )


def test_empty_stroke_and_consumed_fill_stay_invisible():
    scene = _scene()
    with a.Off():
        line = a.Line(
            color=a.WHITE, glow=0.5, cap_style="square", stroke_width=0
        ).spawn()
    assert not _frames(scene).any()
    with a.Off():
        line.despawn()
        a.Square(
            color=a.RED, stroke_color=a.GREEN, joint_type="miter", stroke_width=200
        ).spawn()
    frame = _frames(scene)[0]
    assert frame[..., 1].max() > frame[..., 0].max()
    assert frame[48, 64, 1] > frame[48, 64, 0]


@pytest.mark.parametrize("samples", [1, 4])
def test_caps_in_supersampled_and_path_traced_renders(samples):
    a.SETTINGS.raytracing.experimental.analytic_aa = False
    a.SETTINGS.raytracing.samples_per_pixel = samples
    scene = _scene()
    scene.video_settings.set(supersampling=2)
    with a.Off():
        line = a.Line(stroke_width=65, cap_style="butt", color=a.WHITE).spawn()
    butt = _frames(scene)[0]
    line.cap_style = "square"
    square = _frames(scene)[0]
    assert (square[..., 0] > 128).sum() > (butt[..., 0] > 128).sum()


def test_nonplanar_strokes_diagnose_unsupported_styles():
    # A spatial cubic cannot use a single plane for its offset outline.
    curve = a.BezierCurveCubic(
        [[-1, 0, 0], [0, 1, 1], [0, -1, 1], [1, 0, 0]], cap_style="butt"
    )
    with pytest.raises(a.AlganConfigurationError, match="unshaded planar path"):
        curve.get_render_primitives()


def test_moving_off_axis_keeps_the_stroke_width():
    scene = _scene()
    with a.Off():
        line = a.Line(stroke_width=50, cap_style="square", color=a.WHITE).spawn()
    centered = _frames(scene)[0, ..., 0] > 128
    with a.Off():
        line.move(a.RIGHT * 4)
    offset = _frames(scene)[0, ..., 0] > 128
    assert int(centered.sum(0).max()) == int(offset.sum(0).max())
