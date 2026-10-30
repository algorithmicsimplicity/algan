"""Batches the sheet route used to refuse: camera clip planes, and a
transparent background under an environment map.

``analytic_raster_route_active`` vetoed ``camera.near > 0``, because near
clipping lived only in ray traversal, and a transparent background together
with an environment map, because the env prefill left the alpha column to the
background. Both sent the whole batch to the classic supersampled wavefront.
The emission now cuts geometry at the near plane and culls what lies wholly
past either clip plane, and the env prefill writes its alpha opaque, so these
batches stay on the analytic route. These tests pin the emission's clip
flags, and render each feature on both routes and compare.

Feature tests of the renderer: not ``fast``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan import (
    BLUE,
    GREEN,
    RED,
    RIGHT,
    SETTINGS,
    TRANSPARENT,
    UP,
    WHITE,
    Cube,
    Off,
    Scene,
    Sphere,
    Square,
    VideoSettings,
)
from algan.rendering.raytracing import raster_pipeline, tracer
from algan.scene_manager import SceneManager

RES = (96, 54)


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.fixture
def routes(monkeypatch):
    """Every route decision the renders below made."""
    decisions = []
    real = tracer.analytic_raster_route_active

    def spy(*args, **kwargs):
        decision = real(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(tracer, "analytic_raster_route_active", spy)
    return decisions


def _frame(build, *, sheet=True, near=0.0, far=0.0, background=WHITE, env=None):
    """One frame of ``build``'s scene: the sheet route at output resolution, or
    the classic wavefront at 4x4 supersampling as the reference.
    """
    SETTINGS.raytracing.set(shadows=False, samples_per_pixel=1)
    SETTINGS.raytracing.experimental.hybrid_raster = sheet
    video = VideoSettings(
        resolution=RES, frames_per_second=10, supersampling=1 if sheet else 4
    )
    with Scene(video_settings=video) as scene:
        scene.set_background(background)
        if env is not None:
            scene.set_environment_map(env)
        with Off():
            build()
        if near:
            scene.camera.near = near
        if far:
            scene.camera.far = far
        return torch.cat([f.cpu() for f in scene.get_frames(0, 1)]).float()[0]


def _off(a, b):
    """Per-pixel largest channel difference over RGB."""
    return (a[..., :3] - b[..., :3]).abs().amax(-1)


# ---- The emission's clip flags ---------------------------------------------


def _flags(z_ranges, near, far):
    """Column 9 of the projection table for triangles spanning ``z_ranges``,
    seen by a camera at the origin looking down +z.
    """
    verts = torch.tensor(
        [[[0.0, 0.0, z0], [1.0, 0.0, z1], [0.0, 1.0, z1]] for z0, z1 in z_ranges]
    )[None]
    ro = torch.zeros(1, 3)
    sp = torch.tensor([[0.0, 0.0, 1.0]])
    pbx = torch.tensor([[0.01, 0.0, 0.0]])
    pby = torch.tensor([[0.0, 0.01, 0.0]])
    packed = raster_pipeline._triangle_projection_fused(
        verts,
        ro,
        sp,
        pbx,
        pby,
        48.0,
        27.0,
        False,
        None,
        raster_pipeline._clip_planes(near, far, verts.device),
    )
    return packed[0, :, 9].tolist()


def test_triangles_are_flagged_against_the_camera_clip_planes():
    """-1 culls a triangle wholly nearer than the near plane or wholly past
    the far clip; 2 sends an all-front triangle the near plane crosses to the
    exact screen-space cut; a camera-plane straddler keeps its ray cast (0),
    whose samples test the plane themselves.
    """
    spans = [(5, 6), (1, 2), (2, 4), (12, 14), (8, 12), (-1, 5)]
    assert _flags(spans, 3.0, 10.0) == [1.0, -1.0, 2.0, -1.0, 1.0, 0.0]
    # Each plane alone.
    assert _flags(spans, 3.0, 0.0) == [1.0, -1.0, 2.0, 1.0, 1.0, 0.0]
    assert _flags(spans, 0.0, 10.0) == [1.0, 1.0, 1.0, -1.0, 1.0, 0.0]
    # No clip planes, no clip argument: the projection a batch always ran.
    assert raster_pipeline._clip_planes(0.0, 0.0, torch.device("cpu")) is None
    assert _flags(spans, 0.0, 0.0) == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_point_sets_wholly_outside_a_clip_plane_are_culled():
    """The circuits' screen-bounds cull: a bounding box is out only when every
    corner is past the same plane.
    """
    ro = torch.zeros(1, 3)
    sp = torch.tensor([[0.0, 0.0, 1.0]])

    def boxes(*zs):
        return torch.tensor([[[0.0, 0.0, z] for z in group] for group in zs])

    points = boxes((1.0, 2.0), (2.0, 4.0), (5.0, 6.0), (9.0, 11.0), (11.0, 12.0))
    out = raster_pipeline._outside_clip_planes(points, ro, sp, 3.0, 10.0)
    assert out.tolist() == [True, False, False, False, True]
    assert raster_pipeline._outside_clip_planes(points, ro, sp, 0.0, 0.0) is None


# ---- Renders -----------------------------------------------------------------


def _env_bytes():
    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(20, 230, 32, dtype=np.uint8)[None, :]
    img[..., 1] = 90
    img[..., 2] = np.linspace(200, 30, 16, dtype=np.uint8)[:, None]
    return torch.from_numpy(img)


def _ball_and_pane():
    Sphere(radius=0.6).spawn(animate=False)
    pane = Square(size=0.8, color=RED, fill_opacity=0.5)
    pane.move_to((-1.5, 0.0, 0.0))
    pane.spawn(animate=False)


def test_a_transparent_background_under_an_env_map_stays_on_the_sheet_route(routes):
    """The map replaces the backdrop on every route, so its frame is opaque:
    alpha 255 everywhere, and the colour the same as over an opaque
    background. The env prefill used to leave the alpha column to the
    transparent background, which is why the route refused the batch.
    """
    transparent = _frame(_ball_and_pane, background=TRANSPARENT, env=_env_bytes())
    assert set(routes) == {True}
    assert transparent.shape[-1] == 4
    assert bool((transparent[..., 3] == 255).all())
    opaque = _frame(_ball_and_pane, env=_env_bytes())
    assert float(_off(transparent, opaque).max()) <= 1


def _cut_sphere():
    Sphere(radius=3, color=BLUE).spawn(animate=False)


def test_the_near_plane_opens_a_solid_on_the_sheet_route(routes):
    """The near plane cuts the front off an opaque sphere, which shows its
    inside; the sheet route draws it as the supersampled classic route does.

    REGRESSION for the cut's rim: the closed-shell opacity ceiling spends a
    solid's far shell only after its near one, assuming every ray crosses it
    twice. Rays through the opening start inside it, so on the pixels the cut
    runs through the far shell was left its area as a fraction of the light
    the near shell had already taken, and the background showed through in a
    light ring (77 pixels off the reference by more than 64 levels here).
    """
    sheet = _frame(_cut_sphere, near=18.0)
    assert set(routes) == {True}
    whole = _frame(_cut_sphere)
    assert int((_off(sheet, whole) > 64).sum()) > 500, "the near plane cut nothing"
    classic = _frame(_cut_sphere, sheet=False, near=18.0)
    assert set(routes) == {True, False}
    off = _off(sheet, classic)
    # What is left is anti-aliasing: the classic reference box-filters 16
    # samples, the sheet route takes exact areas.
    assert int((off > 64).sum()) <= 8
    assert float(off.mean()) < 1.0


def _translucent_cube():
    cube = Cube(size=5, color=RED)
    cube.rotate(30, UP).rotate(20, RIGHT)
    cube.spawn(animate=False)


def test_the_near_plane_opens_a_translucent_flat_solid(routes):
    """A translucent cube keeps its closed-shell ceiling where the near plane
    cuts it, and a flat-shaded one splits its far shell into one sibling
    sheet per face.

    REGRESSION: on a rim pixel the ceiling spent the far shell's allowance on
    its first face and none on the second, and a band whose closing sibling
    claims nothing used to skip that sibling and with it the band's
    occlusion write -- the first face painted and occluded nothing, and the
    background showed through (a pixel 62 levels off the reference here).
    """
    sheet = _frame(_translucent_cube, near=18.5)
    classic = _frame(_translucent_cube, sheet=False, near=18.5)
    off = _off(sheet, classic)
    assert float(off.max()) <= 48, (
        f"{int((off > 48).sum())} pixels off by up to {float(off.max()):.0f}"
    )


def _three_depths():
    Sphere(radius=2, color=BLUE).move_to((-4, 0, 6)).spawn(animate=False)
    Sphere(radius=2, color=RED).move_to((0, 0, -4)).spawn(animate=False)
    cube = Cube(size=3, color=GREEN, opacity=1)
    cube.move_to((4, 0, -12)).rotate(25, UP)
    cube.spawn(animate=False)
    pane = Square(size=4, color=GREEN, fill_opacity=1, stroke_width=0)
    pane.move_to((5, 2, -15)).spawn(animate=False)
    tilted = Square(size=3, color=RED, fill_opacity=0.8, stroke_width=0)
    tilted.rotate(-70, RIGHT).move_to((-5, -2, -4)).spawn(animate=False)


def test_the_far_clip_culls_at_emission_without_changing_the_picture(
    routes, monkeypatch
):
    """Triangles and circuits wholly past the far clip are culled before any
    fragment exists; the resolve, which stops each pixel's walk at the far
    clip, drew none of them anyway. A sphere and a circuit straddling the
    clip keep their near parts.
    """
    culled = _frame(_three_depths, far=24.0)
    assert set(routes) == {True}
    unclipped = _frame(_three_depths)
    assert int((_off(culled, unclipped) > 32).sum()) > 100, (
        "the far clip removed nothing"
    )
    real_clip = raster_pipeline._clip_planes
    real_outside = raster_pipeline._outside_clip_planes
    monkeypatch.setattr(
        raster_pipeline, "_clip_planes", lambda n, f, d: real_clip(n, 0.0, d)
    )
    monkeypatch.setattr(
        raster_pipeline,
        "_outside_clip_planes",
        lambda p, ro, sp, n, f: real_outside(p, ro, sp, n, 0.0),
    )
    kept = _frame(_three_depths, far=24.0)
    # One level at most: the closed-shell ceiling spends from a float prefix
    # over the whole fragment stream, whose low bits the culled fragments
    # move.
    assert float(_off(culled, kept).max()) <= 1
