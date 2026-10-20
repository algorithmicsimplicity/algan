"""A Mob's glow reaches the renderer once, whatever builds its geometry.

``mob.glow``, plus any glow in the Mob's own colour, is the glow lane bloom
reads. A 2-D shape, a flat triangle Mob, a ``Surface``, a triangle mesh, the PN
soup a ``become`` morphs through and a curved Manim tile each build their render
primitive differently, so each is checked here for the lane it hands the
renderer, and one render shows what that means on screen.

A feature test, so it sits outside the fast suite: it only breaks when one of
these primitive builders changes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

import algan.external_libraries.manim as manim
from algan import (
    OUT,
    RED,
    SMOKE_TEST,
    Color,
    ManimMob,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    Sphere,
    Square,
    Surface,
    Text,
    VideoSettings,
)
from algan.animatable_base.morph_conversions import convert_to_pn_soup
from algan.mobs.shapes_2d import TriangleVertices
from algan.mobs.three_d_models.mesh import TriangleMesh

GLOW = 0.25
HALF = 0.8
CORNERS = [[-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0]]


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _plane(uv):
    u, v = uv[..., :1], uv[..., 1:2]
    return torch.cat(
        [(u - 0.5) * 2 * HALF, (v - 0.5) * 2 * HALF, torch.zeros_like(u)], -1
    )


def _triangles(**kwargs):
    """The square ``_plane`` draws, as two flat triangles."""
    a, b, c, d = CORNERS
    return TriangleVertices(torch.tensor([a, b, c, a, c, d]), **kwargs)


def _manim_tile(**kwargs):
    """One curved quad of a stock Manim sphere, which renders as PN patches."""
    tiles = list(manim.Sphere(resolution=(8, 4)).family_members_with_points())
    tile = ManimMob(tiles[6], **kwargs)
    assert tile._nonplanar_plan.mode == "patch"
    return tile


FLAT = {
    "square": Square,
    "triangles": _triangles,
}
MESHES = {
    "surface": lambda **kwargs: Sphere(radius=0.5, **kwargs),
    "triangle mesh": lambda **kwargs: TriangleMesh(
        torch.tensor(CORNERS), [[0, 1, 2], [0, 2, 3]], **kwargs
    ),
    "pn soup": lambda **kwargs: convert_to_pn_soup(
        Sphere(radius=0.5, **kwargs), **kwargs
    ),
    "manim tile": _manim_tile,
}


def _glow_lane(mob):
    """Every glow value the Mob's render primitives carry where they are drawn."""
    primitives = mob.get_render_primitives()
    if not isinstance(primitives, (list, tuple)):
        primitives = [primitives]
    lanes = []
    for primitive in primitives:
        colors = getattr(primitive, "colors", None)
        if colors is None:
            continue
        colors = colors.as_subclass(torch.Tensor).reshape(-1, colors.shape[-1])
        lanes.append(colors[colors[:, -1] > 0][:, -2])
    lane = torch.cat(lanes)
    assert lane.numel() > 0
    return lane


def _assert_lane(mob, expected):
    lane = _glow_lane(mob)
    torch.testing.assert_close(lane, torch.full_like(lane, expected), atol=1e-6, rtol=0)


@pytest.mark.parametrize("make", FLAT.values(), ids=FLAT.keys())
def test_a_flat_mob_hands_the_renderer_its_glow(scene, make):
    mob = make(add_to_scene=False)
    mob.glow = GLOW
    _assert_lane(mob, GLOW)


@pytest.mark.parametrize("make", MESHES.values(), ids=MESHES.keys())
def test_a_mesh_hands_the_renderer_its_glow_once(scene, make):
    """REGRESSION. Each of these added the glow into its colours and then
    passed that lane on as the primitive's ``glow`` too, which the primitive
    adds to the same lane: a glow of 0.25 reached the renderer as 0.5, so a
    sphere glowed twice as strongly as a square with the same ``glow``.
    """
    mob = make(add_to_scene=False)
    mob.glow = GLOW
    _assert_lane(mob, GLOW)


def test_a_colours_own_glow_counts_once_too(scene):
    """REGRESSION. The doubled lane held the colour's own glow as well as the
    Mob's ``glow``, so both came out twice.
    """
    sphere = Sphere(radius=0.5, color=RED.set_glow(0.1), add_to_scene=False)
    sphere.glow = GLOW
    _assert_lane(sphere, 0.1 + GLOW)


def _frame(tmp_path, make, glow):
    """One SMOKE_TEST still of ``make()``, unlit red, at ``glow``."""
    with Scene() as scene:
        scene.set_video_settings(SMOKE_TEST.set(resolution=(160, 90)))
        mob = make()
        mob.set_material(MeshBasicMaterial(color=RED))
        with Off():
            mob.spawn()
            mob.glow = glow
        scene.wait(1)
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.png"
        scene.save_frame(path, overwrite=True)
    return np.asarray(Image.open(path)).astype(np.int16)[..., :3]


def test_a_glowing_surface_draws_what_flat_triangles_of_its_shape_draw(tmp_path):
    """REGRESSION. A flat ``Surface`` and the same square built from two
    triangles drew the same until they glowed; then the surface's halo, and
    the frame's total brightness, came out about twice the triangles'.
    """

    def surface():
        return Surface(_plane, grid_width=4, grid_height=4)

    triangles = _frame(tmp_path, _triangles, 0.5)
    # The halo is most of what is drawn, so a doubled glow cannot hide in it.
    assert triangles.sum() > 3 * _frame(tmp_path, _triangles, 0.0).sum()
    assert int(np.abs(_frame(tmp_path, surface, 0.5) - triangles).max()) <= 2


def _studio_band(height=128, width=256):
    """A bright horizon band behind the camera: what chrome text reflects."""
    lat = torch.linspace(90, -90, height).view(height, 1).expand(height, width)
    lon = torch.linspace(0, 360, width).view(1, width).expand(height, width)
    behind = (lon - 270 + 180) % 360 - 180
    band = (
        torch.sigmoid((lat + 5) / 1.2)
        * torch.sigmoid((5.5 - lat) / 1.2)
        * torch.exp(-((behind / 80) ** 2))
        * 1.3
    )
    return band.unsqueeze(-1).expand(height, width, 3).contiguous()


def _chrome_behind_a_faded_title(tmp_path, with_title):
    """Chrome text under a studio band, seen from above, with or without a
    glowing two-line title in front of it that has faded to zero opacity.
    """
    with Scene() as scene:
        scene.set_video_settings(VideoSettings(resolution=(192, 108), supersampling=1))
        scene.set_environment_map(_studio_band(), intensity=1.0, ambient=True)
        chrome = Text("Algan", font_size=200)
        chrome.set_material(
            MeshStandardMaterial(color=Color("#eef1f5"), metalness=1.0, roughness=0.08)
        )
        with Off():
            chrome.scale_to_height(3.0).spawn()
            if with_title:
                title = Text(
                    "Algorithmic\nAnimation",
                    font_size=110,
                    color=Color("#4da3ff").set_glow(0.15).set_opacity(0.0),
                )
                title.scale_to_width(12.8).move(OUT * 0.03).spawn()
            camera = scene.get_camera()
            camera.move_to(torch.tensor([-0.3, 2.3, 24.5]))
            camera.look_at(torch.tensor([-0.3, -0.45, 0.0]))
        path = tmp_path / f"{int(with_title)}.png"
        scene.save_frame(path, overwrite=True)
    return np.asarray(Image.open(path)).astype(np.int16)[..., :3]


def test_a_faded_out_glowing_title_leaves_the_chrome_behind_it_alone(tmp_path):
    """REGRESSION. A circuit kept its sheet while it had glow, whatever its
    opacity, and an invisible sheet in front of a glossy reflector still
    changed how the reflector shaded. The demo video's faded title left ghost
    letters in the chrome wordmark behind it, up to 36 levels deep. Glow is
    weighted by alpha on every route, so a circuit at zero opacity is absent.
    """
    alone = _chrome_behind_a_faded_title(tmp_path, with_title=False)
    # The scene has to show the chrome, or the comparison proves nothing.
    assert alone.max() > 120
    faded = _chrome_behind_a_faded_title(tmp_path, with_title=True)
    assert int(np.abs(faded - alone).max()) <= 2
