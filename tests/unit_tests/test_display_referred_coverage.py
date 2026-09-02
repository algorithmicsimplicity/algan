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
area average of the *displayed* colour. The statement of that is a supersampled
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

import cv2
import numpy as np
import pytest

from algan import (
    DARKER_GRAY,
    OUT,
    WHITE,
    AmbientLight,
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


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, None)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return out * 255.0


def _render(tmp_path, name, side):
    """One frame of the probe scene, as an (H, W, 3) uint8 RGB array."""
    SceneManager.instance().reset()
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=_AMBIENT).spawn(animate=False)
        # A slab facing the camera, turned in its own plane so its four edges
        # cross pixels at every sub-pixel phase. Flat triangles, so the 6x
        # render sees the SAME geometry -- diced geometry would tessellate
        # finer there and the two renders would then disagree about the shape
        # rather than about the resolve.
        slab = Prism(width=3.2, height=3.2, depth=0.2).set_material(
            MeshLambertMaterial(color=_COLOR)
        )
        slab.rotate(19, OUT)
        slab.spawn(animate=False)
    out = tmp_path / f"{name}.png"
    Scene.save_frame(str(out), VideoSettings((side, side), 30, supersampling=1))
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert im is not None, f"the probe render produced no file at {out}"
    return im[..., 2::-1].astype(np.float64)


def _silhouette(frame):
    """Pixels the solid shares with the background: the resolve's own case.

    A pixel is on the silhouette when it is not pure background itself and one
    of its four neighbours is. The probe scene is a single flat slab so this is
    all of its partial coverage; a faceted solid would also have interior
    creases, and those are deliberately out of scope -- two faces that
    partition a pixel exactly leave no geometric residual, the composite has
    nothing to clamp against, and their over-range sum is still blended in
    linear (DESIGN_sheet_resolve.md ss4.8's closing paragraph).
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
