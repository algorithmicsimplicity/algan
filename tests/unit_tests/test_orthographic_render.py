"""The orthographic camera, rendered.

``test_camera_projection.py`` holds the camera's side of
``Camera.set_orthographic``; these render it. Every route used to take the eye
as the one point all rays leave from -- ray generation, pixel footprints, the
raster tables, stroke widths, the view vector a surface is shaded with -- so
each is checked in pixels against the projection's two promises:

* depth changes nothing on screen: the same shapes 12 units nearer or farther
  render the same image, filled and stroked shapes, a tessellated sphere, a
  camera-facing 3-D stroke and Manim's curved patches alike;
* the frame on the plane through ``ORIGIN`` is the perspective camera's, so
  switching leaves flat shapes there exactly where they were.

Both hold on the sheet route, the classic wavefront tracer and the path tracer.
Then the compositing passes, which measure depth from the camera's plane and
clip at its near plane there, and a ``CameraView`` inset of an orthographic
view, which is orthographic too.

Unmarked, so outside ``--fast``: only the projection's own code can break them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import algan.manim as mn
from algan import (
    BLACK,
    BLUE,
    DOWN,
    GREEN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    YELLOW,
    CameraView,
    Circle,
    DirectionalLight,
    Off,
    Scene,
    Sphere,
    Square,
    VideoSettings,
)
from algan.rendering import camera_views
from algan.rendering.raytracing import aux_passes
from algan.scene_manager import SceneManager

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)
ROUTES = {
    "sheet": {"samples": 1, "experimental": {}},
    "classic wavefront": {"samples": 1, "experimental": {"hybrid_raster": False}},
    "path tracer": {"samples": 4, "experimental": {}},
}


def _shapes(depth):
    """One of each kind of geometry, ``depth`` units out of the origin plane."""
    out = OUT * depth
    Square(size=2, color=RED).move(LEFT * 4 + UP * 1.5 + out).spawn()
    Circle(radius=1, color=GREEN).move(UP * 1.5 + out).spawn()
    Sphere(radius=1, color=BLUE).move(RIGHT * 4 + UP * 1.5 + out).spawn()
    # A non-planar stroke, drawn as a camera-facing ribbon, and Manim's
    # curved sphere tiles, drawn as patches (algan.mobs.nonplanar_circuit).
    mn.ParametricFunction(
        lambda t: np.array([0.8 * np.cos(t), 0.8 * np.sin(t), t / 12]),
        t_range=[0, 4 * np.pi],
        stroke_width=4,
        color=YELLOW,
    ).move_to(LEFT * 3 + DOWN * 2 + out).spawn()
    mn.Sphere(radius=1, resolution=(8, 4)).move_to(RIGHT * 3 + DOWN * 2 + out).spawn()


def _flat_shapes():
    """Flat shapes on the origin plane, where both projections agree."""
    Square(size=2, color=RED).move(LEFT * 4 + UP * 1.5).spawn()
    Circle(radius=1, color=GREEN).move(UP * 1.5).spawn()
    Square(size=3, color=BLUE).rotate(30, OUT).move(RIGHT * 3 + DOWN * 2).spawn()


def _render(route, build, *, orthographic=True, camera=None, passes=False):
    """One frame of the scene ``build()`` makes, rendered by ``route``.

    One directional light, which shades a surface alike wherever it is, so
    only the projection can tell two depths apart. ``camera`` holds camera
    attributes to set. Returns the frame's ``[H, W, 3]`` pixels, and with
    ``passes`` the auxiliary passes beside them.
    """
    config = ROUTES[route]
    SETTINGS.raytracing.set(samples_per_pixel=config["samples"], denoise=False)
    for name, value in config["experimental"].items():
        setattr(SETTINGS.raytracing.experimental, name, value)
    SceneManager.reset()
    auxes = []
    try:
        with Scene(video_settings=VIDEO) as scene:
            scene.set_background(BLACK)
            with Off():
                Scene.clear_lights()
                DirectionalLight(
                    location=OUT * 10 + UP * 5 + LEFT * 3, target=ORIGIN
                ).spawn()
                if orthographic:
                    scene.camera.set_orthographic()
                build()
            for name, value in (camera or {}).items():
                setattr(scene.camera, name, value)
            kwargs = {"aux_passes": True, "aux_sink": auxes.append} if passes else {}
            frames = [f.cpu() for f in scene.get_frames(0, 1, **kwargs)]
    finally:
        SceneManager.reset()
    rgb = torch.cat(frames).float()[0, ..., :3]
    return (rgb, auxes[0]) if passes else rgb


def _lit(rgb):
    return rgb.amax(-1) > 8


def _assert_same_image(route, actual, expected):
    """Equal to the repository's tolerance of 2 levels a channel.

    The deterministic routes must agree everywhere. The path tracer's
    samples land at the same places on screen, but a sample on a shape's edge
    can round to the other side of it: a handful of edge pixels may differ.
    """
    off = (actual - expected).abs().amax(-1) > 2
    allowed = 0 if ROUTES[route]["samples"] == 1 else int(0.005 * _lit(expected).sum())
    assert int(off.sum()) <= allowed, (
        f"{int(off.sum())} pixels differ by more than 2 levels, "
        f"by up to {float((actual - expected).abs().max()):.0f}"
    )


@pytest.mark.parametrize("route", list(ROUTES))
def test_depth_changes_nothing_on_screen(route):
    near = _render(route, lambda: _shapes(6.0))
    far = _render(route, lambda: _shapes(-6.0))
    assert int(_lit(near).sum()) > 1500, "the shapes should fill much of the frame"
    _assert_same_image(route, far, near)


@pytest.mark.parametrize("route", list(ROUTES))
def test_the_origin_plane_keeps_its_perspective_frame(route):
    perspective = _render(route, _flat_shapes, orthographic=False)
    orthographic = _render(route, _flat_shapes)
    assert int(_lit(perspective).sum()) > 1500
    _assert_same_image(route, orthographic, perspective)


#: Rows above this hold the sphere of :func:`_depth_shapes`, rows below it the
#: squares.
SPLIT = 40


def _depth_shapes():
    """Two equal squares 5 units nearer and 5 farther than the origin, and a sphere.

    The squares sit at ``x = -2.7`` and ``x = 2.7``, where no pixel centre lies
    on an edge, so the two cover the same number of pixel centres.
    """
    Square(size=2, color=RED, stroke_width=0).move(
        LEFT * 2.7 + DOWN * 1.5 + OUT * 5
    ).spawn()
    Square(size=2, color=BLUE, stroke_width=0).move(
        RIGHT * 2.7 + DOWN * 1.5 + IN * 5
    ).spawn()
    Sphere(radius=1).move(UP * 2.5).spawn()


def test_the_depth_pass_measures_from_the_camera_plane():
    _rgb, aux = _render("sheet", _depth_shapes, passes=True)
    depth = aux["depth"][0]
    hit = aux["mob_id"][0] != aux_passes.MISS_MOB_ID
    near_square, far_square = hit[SPLIT:, :80], hit[SPLIT:, 80:]
    # Equal squares cover equal pixels: 2 units at 90 / 8 pixels a unit, 23
    # pixel centres across and 22 down.
    assert int(near_square.sum()) == int(far_square.sum()) == 23 * 22
    # The camera sits 20 units out, so the squares are 15 and 25 deep at every
    # pixel -- planar depth from the camera's plane, the depth each ray runs.
    near = depth[SPLIT:, :80][near_square]
    far = depth[SPLIT:, 80:][far_square]
    torch.testing.assert_close(near, torch.full_like(near, 15.0), atol=1e-3, rtol=0)
    torch.testing.assert_close(far, torch.full_like(far, 25.0), atol=1e-3, rtol=0)
    # The front of the sphere is a unit nearer than its centre.
    sphere = depth[:SPLIT][hit[:SPLIT]]
    assert float(sphere.min()) == pytest.approx(19.0, abs=0.05)


def test_the_near_plane_is_planar_from_the_camera_plane():
    # Between the two squares: the near one goes everywhere, in the frame and
    # the passes alike. A near plane also sends the frame to the supersampled
    # fallback route.
    rgb, aux = _render("sheet", _depth_shapes, camera={"near": 20.0}, passes=True)
    depth = aux["depth"][0][SPLIT:]
    assert bool(torch.isinf(depth[:, :80]).all())
    assert int(_lit(rgb[SPLIT:, :80]).sum()) == 0
    far_square = torch.isfinite(depth[:, 80:])
    assert int(far_square.sum()) == 23 * 22
    far = depth[:, 80:][far_square]
    torch.testing.assert_close(far, torch.full_like(far, 25.0), atol=1e-3, rtol=0)
    assert int(_lit(rgb[SPLIT:, 80:]).sum()) >= 23 * 22


def test_a_camera_view_of_an_orthographic_view_is_orthographic(monkeypatch):
    captures = []
    real = camera_views._texture_from_capture

    def recording(frames):
        captures.append(frames.float().clone())
        return real(frames)

    monkeypatch.setattr(camera_views, "_texture_from_capture", recording)

    def build():
        for color, place in ((RED, LEFT * 2 + OUT * 6), (BLUE, RIGHT * 2 + IN * 6)):
            Square(size=2, color=color, fill_opacity=1, stroke_width=0).move(
                place
            ).spawn()
        view = CameraView(resolution=(96, 64), height=2)
        assert view.camera.orthographic
        torch.testing.assert_close(
            view.camera.visible_size_at(OUT * 6), view.camera.visible_size_at(IN * 6)
        )
        view.move_to(DOWN * 3).spawn()

    _render("sheet", build)
    (capture,) = captures
    level = capture[0, ..., :3].amax(-1)
    # Mostly covered pixels: the squares' edges fall on pixel edges, so a
    # pixel either side of one is covered fully or hardly at all.
    covered = level > 0.5 * float(level.max())
    # The capture keeps the main view's 8-unit frame height, 8 pixels a unit
    # here: both squares are 16 pixels across although one is 12 units nearer.
    for square in (covered[:, :48], covered[:, 48:]):
        assert int(square.any(0).sum()) == 16
        assert int(square.any(1).sum()) == 16
        assert int(square.sum()) == 16 * 16
