"""A partially covered pixel is resolved the way a supersampled render is.

The composite blends a pixel's parts in linear HDR and the display transform
(``clamp`` then the sRGB OETF) runs afterwards, so a channel whose radiance is
above the display range carries no coverage information at all. At ORANGE under
this package's own example lighting the cone's red is 1.85: a 60%-covered
silhouette pixel still holds 1.11, shows the same 255 as the interior, and only
its green and blue fall with coverage. The fringe changes HUE rather than
fading, which reads as a saturated, darker rim along the silhouette.

``SETTINGS.raytracing.experimental.aa_display_resolve`` clamps the geometry's
premultiplied contribution to the pixel area it covers, which is exactly the
area average of the *displayed* colour: each sheet's to its own claim in the
resolve, and their sum to what the background leaves in the composite. The
statement of that is a supersampled
render: these tests render the same scene at 1x and at 6x, average the 6x down
IN LINEAR LIGHT (averaging encoded values is a different, wrong answer, and one
that would fail here for the pipeline's own gamma rather than for coverage),
and compare. They render because the interaction is between the composite and
the display transform, which sit at opposite ends of the pipeline and cannot be
pinned against each other anywhere in between.

Feature tests of the render boundary, not of anything the timeline or the Scene
can break, so they stay out of the fast suite.
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from algan import (
    DARKER_GRAY,
    OUT,
    UP,
    WHITE,
    AmbientLight,
    DirectionalLight,
    MeshLambertMaterial,
    Off,
    Prism,
    Scene,
    SceneManager,
    VideoSettings,
)
from algan.settings import SETTINGS

#: Small enough to render four times in a test, big enough for a silhouette
#: tens of pixels long at every sub-pixel phase.
_SIDE = 96
_SS = 6

#: Ambient alone, so the slab is lit uniformly and every silhouette pixel is a
#: clean coverage ramp of ONE colour rather than a shading gradient as well.
#: 2.0 of it takes the authored red to 2.0 in linear light and leaves green and
#: blue inside the display range, which is the whole point: under a linear
#: resolve red is pinned at 255 across the ramp while the other two fade.
_AMBIENT = 2.0
_COLOR = (1.0, 0.45, 0.2)
#: A matte board for the slab to stand in front of: inside the display range
#: under the same light, so only the slab's edge is over range.
_BOARD = (0.3, 0.3, 0.3)


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return out * 255.0


def _slab(glow=0.0):
    """The probe scene: a slab facing the camera under the ambient light.

    Turned in its own plane so its four edges cross pixels at every sub-pixel
    phase. Flat triangles, so the 6x render sees the SAME geometry -- diced
    geometry would tessellate finer there and the two renders would then
    disagree about the shape rather than about the resolve.
    """
    AmbientLight(color=WHITE, intensity=_AMBIENT).spawn(animate=False)
    slab = Prism(width=3.2, height=3.2, depth=0.2).set_material(
        MeshLambertMaterial(color=_COLOR)
    )
    slab.rotate(19, OUT)
    if glow:
        slab.glow = glow
    slab.spawn(animate=False)


def _slab_before_a_board(width=10.0, x=5.0, glow=0.0):
    """The slab with a board behind it, by default behind its right half only.

    Where the slab's edge is over the board, the two sheets fill the pixel
    between them, so the composite has no background residual to clamp
    against: only the resolve's clamp of each sheet reaches that edge.
    """
    _slab(glow)
    board = Prism(width=width, height=width, depth=0.1).set_material(
        MeshLambertMaterial(color=_BOARD)
    )
    board.move_to((x, 0.0, -1.0))
    board.spawn(animate=False)


def _cube():
    """A cube turned to show two faces: one lit past the display range in red
    (1.4), the other inside it (0.47).

    The crease between them is the shading split's two band siblings, one
    mesh's sheets partitioning each pixel they share (DESIGN_sheet_resolve.md
    ss4.7), so it too leaves the composite nothing to clamp against. Turned in
    the image plane as well, so the crease crosses pixels at every phase.
    """
    AmbientLight(color=WHITE, intensity=0.5).spawn(animate=False)
    DirectionalLight(color=WHITE, intensity=1.25, location=(4.0, 0.0, 2.0)).spawn(
        animate=False
    )
    cube = Prism(width=2.4, height=2.4, depth=2.4).set_material(
        MeshLambertMaterial(color=_COLOR)
    )
    cube.rotate(35, UP)
    cube.rotate(12, OUT)
    cube.spawn(animate=False)


def _render(tmp_path, name, side, build=_slab):
    """One frame of a probe scene, as an (H, W, 3) uint8 RGB array."""
    SceneManager.instance().reset()
    Scene.set_background(DARKER_GRAY)
    with Off():
        build()
    out = tmp_path / f"{name}.png"
    Scene.save_frame(str(out), VideoSettings((side, side), 30, supersampling=1))
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert im is not None, f"the probe render produced no file at {out}"
    return im[..., 2::-1].astype(np.float64)


def _silhouette(frame):
    """Pixels the solid shares with the background: the resolve's own case.

    A pixel is on the silhouette when it is not pure background itself and one
    of its four neighbours is. The probe scene is a single flat slab so this is
    all of its partial coverage. An edge over other geometry, or a crease,
    leaves no background residual at all; :func:`_edge_deviations` measures
    those.
    """
    background = frame[0, 0]
    empty = np.abs(frame - background).max(-1) <= 1
    touches = np.zeros_like(empty)
    touches[1:, :] |= empty[:-1, :]
    touches[:-1, :] |= empty[1:, :]
    touches[:, 1:] |= empty[:, :-1]
    touches[:, :-1] |= empty[:, 1:]
    return touches & ~empty


def _deviation_from_supersampled(tmp_path, tag):
    """Deviation of the 1x render from the 6x reference, along the silhouette.

    Per channel, out of 255.
    """
    one = _render(tmp_path, f"{tag}_1x", _SIDE)
    many = _render(tmp_path, f"{tag}_{_SS}x", _SIDE * _SS)
    lin = _srgb_to_linear(many)
    lin = lin.reshape(_SIDE, _SS, _SIDE, _SS, 3).mean((1, 3))
    reference = _linear_to_srgb(lin)
    edge = _silhouette(one)
    assert edge.sum() > 100, "the probe scene has no silhouette to measure"
    return np.abs(one - reference).max(-1)[edge].max()


def _edge_deviations(tmp_path, tag, build):
    """Deviation of the 1x render from the 6x reference over every partially
    covered pixel: ``(over the background, over other geometry)``.

    Per channel, out of 255. A pixel is partially covered when the 6x render
    does not draw it in one colour (the dither moves a flat area by one level
    at most), and it is over the background when one of its 36 samples is pure
    background -- a crease's pixels are over geometry.
    """
    one = _render(tmp_path, f"{tag}_1x", _SIDE, build)
    many = _render(tmp_path, f"{tag}_{_SS}x", _SIDE * _SS, build)
    blocks = many.reshape(_SIDE, _SS, _SIDE, _SS, 3)
    reference = _linear_to_srgb(_srgb_to_linear(blocks).mean((1, 3)))
    mixed = (blocks.max((1, 3)) - blocks.min((1, 3))).max(-1) > 4
    background = (np.abs(blocks - one[0, 0]).max(-1) <= 1).any((1, 3))
    over_geometry = mixed & ~background
    assert over_geometry.sum() > 40, "the probe scene has no edge over geometry"
    off = np.abs(one - reference).max(-1)
    return off[mixed & background].max(initial=0.0), off[over_geometry].max()


class _aa_display_resolve:  # noqa: N801 - a context manager, not a class API
    """The switch under test, restored to whatever it was before."""

    def __init__(self, enabled):
        self._enabled = enabled

    def __enter__(self):
        experimental = SETTINGS.raytracing.experimental
        self._previous = experimental.aa_display_resolve
        experimental.set(aa_display_resolve=self._enabled)
        return self

    def __exit__(self, *exc):
        SETTINGS.raytracing.experimental.set(aa_display_resolve=self._previous)
        return False


@contextmanager
def _raytracing(**values):
    """Raytracing settings for a render or two, restored afterwards."""
    previous = {name: getattr(SETTINGS.raytracing, name) for name in values}
    SETTINGS.raytracing.set(**values)
    try:
        yield
    finally:
        SETTINGS.raytracing.set(**previous)


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_partial_coverage_matches_a_supersampled_render(tmp_path):
    """The shipped default reproduces what supersampling answers.

    The tolerance is per channel out of 255, and generous: measured on CUDA
    the largest silhouette deviation is under 2, which is the ordinary
    disagreement between a 1x and a 6x rasterization of the same edges.
    """
    assert _deviation_from_supersampled(tmp_path, "on") <= 6


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_the_linear_resolve_alone_does_not(tmp_path):
    """And the test can see the difference it makes.

    Without the clamp the over-range channel is pinned at 255 across the whole
    silhouette ramp, which no supersampled render agrees with. Asserted as a
    gap rather than a number: the point is
    that the switch is load-bearing, not that the broken value is any
    particular one.
    """
    with _aa_display_resolve(False):
        without = _deviation_from_supersampled(tmp_path, "off")
    assert without > 40  # measured 85


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_an_edge_over_other_geometry_matches_a_supersampled_render(tmp_path):
    """REGRESSION. Over the board the slab's red clipped at 255 where a 6x
    render shows 213, 42 levels off: the composite's clamp is against the
    background's residual, and two sheets that fill the pixel leave none. The
    resolve now clamps each sheet to its own claim.
    """
    over_background, over_geometry = _edge_deviations(
        tmp_path, "board", _slab_before_a_board
    )
    assert over_geometry <= 6  # measured 1.0, and 42 before
    assert over_background <= 6


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_a_crease_matches_a_supersampled_render(tmp_path):
    """REGRESSION. The residual DESIGN_sheet_resolve.md ss4.8 left open: a
    crease between a face past the display range and one inside it blended
    the two in linear, 25 levels off a 6x render.
    """
    over_background, over_geometry = _edge_deviations(tmp_path, "crease", _cube)
    assert over_geometry <= 6  # measured 0.9, and 25 before
    assert over_background <= 6  # measured 4.2, and 8.7 before


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_inside_the_display_range_the_clamp_changes_nothing(tmp_path):
    """REGRESSION. At an exposure that brings every colour inside the display
    range there is nothing to clamp. The composite's ceiling still ended at 1,
    whatever the exposure, and it applied to every covered pixel, not just
    the ones with a background residual: at 0.4 the slab's red, 2.0, showed
    as 0.4 instead of 0.8, inside it as well as along its edges.
    """
    with _raytracing(tonemap_exposure=0.4):
        on = _render(tmp_path, "dim_on", _SIDE, _slab_before_a_board)
        with _aa_display_resolve(False):
            off = _render(tmp_path, "dim_off", _SIDE, _slab_before_a_board)
    assert np.array_equal(on, off)


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_the_display_range_follows_the_exposure(tmp_path):
    """The display shows ``clamp(exposure * value)``, so its range ends at
    ``1 / exposure``: at 2.0 both of the slab's edges clamp at half the
    radiance they would at 1.
    """
    with _raytracing(tonemap_exposure=2.0):
        over_background, over_geometry = _edge_deviations(
            tmp_path, "bright", _slab_before_a_board
        )
    assert over_background <= 6
    assert over_geometry <= 6


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_a_glowing_sheet_keeps_its_radiance(tmp_path):
    """Bloom reads a glowing pixel's radiance (``color * glow ** 3``), so the
    resolve leaves a glowing sheet unclamped. With the board behind the whole
    frame no pixel has a background residual and the composite clamps nothing,
    so only the resolve could change this frame, and it does not.

    REGRESSION: the composite clamped every covered pixel to 1, which halved
    the radiance this slab's halo is drawn from.
    """

    def glowing():
        _slab_before_a_board(width=30.0, x=0.0, glow=0.5)

    on = _render(tmp_path, "glow_on", _SIDE, glowing)
    with _aa_display_resolve(False):
        off = _render(tmp_path, "glow_off", _SIDE, glowing)
    assert np.array_equal(on, off)


@pytest.mark.skipif(
    not SETTINGS.raytracing.is_post_process_tonemap_enabled(),
    reason="the in-kernel tonemap composites in display-referred values already",
)
def test_a_tone_curve_keeps_the_linear_blend(tmp_path):
    """Under a tone curve the switch changes nothing.

    A curve compresses an over-range channel instead of clipping it, so the
    channel keeps its coverage, and PBR Neutral moves the other channels with
    the peak: clamping one channel to the covered area moved the other two.

    REGRESSION: the composite also held every covered pixel at 1 before the
    curve, which took the inside of the slab from (249, 127, 85) to
    (240, 131, 46).
    """
    with _raytracing(tonemapping=True, tonemap_method="neutral"):
        on = _render(tmp_path, "curve_on", _SIDE, _slab_before_a_board)
        with _aa_display_resolve(False):
            off = _render(tmp_path, "curve_off", _SIDE, _slab_before_a_board)
    assert np.array_equal(on, off)
