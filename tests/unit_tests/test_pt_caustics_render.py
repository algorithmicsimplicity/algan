"""Renders through the path tracer's MNEE caustics (``pt_caustics``).

What the caustic block (``pt_caustics_taichi``) and the coverage test
(``pt_caustic_cover``) deliver together: lossless glass in a white furnace
must not change it -- the light sample's chain and the BSDF path that makes
the same journey share each such path by MIS, so the light is neither lost
nor counted twice -- and a delta light focuses through glass as optics says
it does, which no BSDF path can show.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan import (
    BLACK,
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    PointLight,
    Prism,
    Scene,
    Sphere,
)

FURNACE = 255 * 0.25


def _glass():
    return MeshPhysicalMaterial(color=WHITE, transmission=1, roughness=0.0, ior=1.5)


def _furnace(tmp_path, name, build, samples=64):
    from test_path_tracer import _furnace_opts, _render_scene_result

    opts = _furnace_opts(True, 16)
    opts["experimental"] = {**opts["experimental"], "pt_caustics": True}
    return _render_scene_result(tmp_path, name, build, samples, **opts)


def _assert_no_caustic_truncation(result):
    truncations = result.render_plan.truncations
    assert truncations.caustic_chain == 0, truncations
    assert truncations.caustic_solver == 0, truncations


def test_a_glass_slab_leaves_the_environment_furnace_unchanged(tmp_path):
    """A floor under a lossless glass slab, in a uniform environment, reads
    the environment: each light sample's chain through the slab and each BSDF
    path that refracts through it to the sky share their paths by the power
    heuristic. Unweighted chains on top of the BSDF paths would brighten the
    floor; chains that culled every BSDF path through the slab would darken it
    wherever the walk misses one.
    """
    from test_path_tracer import _center_patch_mean, _floor_under_glass

    image, result = _furnace(tmp_path, "slab_env.png", _floor_under_glass(1.5))
    assert _center_patch_mean(image, half=4) == pytest.approx(FURNACE, abs=2.5)
    _assert_no_caustic_truncation(result)


def _emissive_box(scene, half=5.0, top=9.0, radiance=0.25):
    """Black walls and a lid emitting ``radiance`` inward, closing a furnace
    over the floor (the walls sink into it and overlap at the corners).
    """
    wall = MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=radiance)
    span = 2 * half + 0.4
    for direction in (LEFT, RIGHT):
        panel = Prism(width=0.2, height=span, depth=top)
        panel.move_to(direction * half + OUT * (top / 2))
        panel.set_material(wall)
        panel.spawn(animate=False)
    for direction in (UP, DOWN):
        panel = Prism(width=span, height=0.2, depth=top)
        panel.move_to(direction * half + OUT * (top / 2))
        panel.set_material(wall)
        panel.spawn(animate=False)
    lid = Prism(width=span, height=span, depth=0.2)
    lid.move_to(OUT * top)
    lid.set_material(wall)
    lid.spawn(animate=False)


def test_a_glass_slab_leaves_an_emitter_furnace_unchanged(tmp_path):
    """The emissive-triangle arm of the same balance: a box of emitters the
    next-event table samples, a white floor and a glass slab between them,
    seen from inside the box.
    """
    from test_path_tracer import _center_patch_mean

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        floor = Prism(width=40.0, height=40.0, depth=0.2)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        _emissive_box(scene)
        slab = Prism(width=9.0, height=9.0, depth=0.5)
        slab.move_to(OUT * 1.0)
        slab.set_material(_glass())
        slab.spawn(animate=False)
        scene.camera.move_to(OUT * 7.0)

    image, result = _furnace(tmp_path, "slab_emitters.png", build)
    assert _center_patch_mean(image, half=4) == pytest.approx(FURNACE, abs=2.5)
    # A rare walk runs out of steps here (about one in a render). An area
    # light's share of that path then arrives through the BSDF path at full
    # weight, which is what the balance above checks, so no zero is asked of
    # ``caustic_solver``.
    assert result.render_plan.truncations.caustic_chain == 0


def test_a_glass_ball_resting_on_the_floor_leaves_the_furnace_unchanged(tmp_path):
    """The hard case for the walk: around a ball's contact point the light
    the ball bends onto the floor comes from chains the straight seed does not
    reach, so those BSDF paths must keep their light (the coverage test), and
    the sky the ball spreads over a wide cone there is the BSDF path's to
    find (MIS). Culling every BSDF path through the ball darkened the floor
    near contact by a third; culling only the covered ones without MIS left
    single pixels at eighteen times the furnace.
    """

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        scene.set_environment_map(torch.full((4, 8, 3), 0.25), ambient=False)
        floor = Prism(width=40.0, height=40.0, depth=0.2)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        ball = Sphere(radius=1.0)
        ball.move_to(OUT * 1.1)
        ball.set_material(_glass())
        ball.spawn(animate=False)

    image, _result = _furnace(tmp_path, "ball_env.png", build)
    rgb = image[..., :3].double()
    assert float(rgb.mean()) == pytest.approx(FURNACE, abs=1.5)
    # The pixels that see the floor through the ball, near the contact point.
    h, w = rgb.shape[:2]
    center = rgb[h // 2 - 3 : h // 2 + 3, w // 2 - 3 : w // 2 + 3]
    assert float(center.mean()) == pytest.approx(FURNACE, abs=4.0)
    assert float(rgb.mean(-1).max()) < 1.5 * FURNACE


# A point light above a raised glass plate, the camera between the plate and
# the floor (so it sees the floor directly; it starts above both), and no
# bounces: what lights the floor is the light sample alone.
_FLOOR_TOP = 0.1
_PLATE_BOTTOM, _PLATE_TOP = 8.0, 12.0
_LIGHT_Z = 13.0


def _point_light_over_plate(plate):
    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        floor = Prism(width=40.0, height=40.0, depth=2 * _FLOOR_TOP)
        floor.set_material(MeshLambertMaterial(color=WHITE))
        floor.spawn(animate=False)
        h = _LIGHT_Z - _FLOOR_TOP
        PointLight(
            location=OUT * _LIGHT_Z,
            color=WHITE,
            intensity=0.4 * math.pi * h * h,
            decay=2,
        ).spawn(animate=False)
        if plate:
            glass = Prism(width=12.0, height=12.0, depth=_PLATE_TOP - _PLATE_BOTTOM)
            glass.move_to(OUT * ((_PLATE_TOP + _PLATE_BOTTOM) / 2))
            glass.set_material(_glass())
            glass.spawn(animate=False)
        scene.camera.move_to(OUT * 7.0)

    return build


@pytest.mark.parametrize("caustics", [True, False])
def test_a_point_light_through_a_glass_plate_lights_the_floor_as_optics_says(
    tmp_path, caustics
):
    """A plate of thickness ``t`` and index ``n`` between a point light and
    the floor, ``h`` below the light, brings the light's image ``t (1 - 1/n)``
    closer: the floor straight below it receives ``T^2 (h / h')^2`` of what it
    receives without the plate, ``h' = h - t (1 - 1/n)`` and ``T`` the
    normal-incidence Fresnel transmittance of each face -- 1.146 here. The
    straight shadow ray of a renderer without caustics passes the plate tinted
    but unbent, so it reads the transmittance alone.
    """
    from test_path_tracer import (
        SMALL_SETTINGS,
        _center_patch_mean,
        _render_scene_result,
    )

    opts = {
        "video": SMALL_SETTINGS,
        "max_bounces": 0,
        "shadows": True,
        "linear_color_space": False,
        "tonemapping": False,
        "experimental": {
            "post_process_tonemap": False,
            "pt_firefly_clamp": 0,
            "pt_error_target": 0,
            "pt_caustics": caustics,
        },
    }
    bare, _ = _render_scene_result(
        tmp_path, "bare.png", _point_light_over_plate(False), 8, **opts
    )
    lit, result = _render_scene_result(
        tmp_path, "plate.png", _point_light_over_plate(True), 8, **opts
    )
    ratio = _center_patch_mean(lit, half=4) / _center_patch_mean(bare, half=4)
    n, t, h = 1.5, _PLATE_TOP - _PLATE_BOTTOM, _LIGHT_Z - _FLOOR_TOP
    transmittance = (1 - ((n - 1) / (n + 1)) ** 2) ** 2
    if caustics:
        expected = transmittance * (h / (h - t * (1 - 1 / n))) ** 2
        assert ratio == pytest.approx(expected, rel=0.02)
        _assert_no_caustic_truncation(result)
    else:
        assert ratio < 1.0


@pytest.mark.parametrize("caustics", [True, False])
def test_a_point_light_reaches_inside_a_subsurface_block(tmp_path, caustics):
    """A scattering interior is reached from a delta light only through its
    refracting boundary, which the straight shadow connection does not cross.
    Without caustics the inside of a subsurface block under a point light
    therefore stays dark; with them each scatter point's light sample
    connects through the top face along the bent path, as a surface point's
    does through glass. Two bounces (in through the face, then the scatter)
    keep it to single scattering, which for a white isotropic medium lit and
    seen from straight above reads

        L = T^2 / n^2 * sigma_s / (4 pi) * I * int_0^d exp(-2 sigma_s z)
            / (h + z / n)^2 dz

    ``T`` the face's normal-incidence transmittance, crossed on the way in and
    out, ``1 / n^2`` the radiance the camera ray gives up entering the glass,
    ``h`` the light's height over the face and ``h + z / n`` the distance the
    bent beam spreads over to depth ``z``. A black floor keeps light that
    passes the block out of the patch.
    """
    from test_path_tracer import _center_patch_mean, _render_scene_result

    n, sigma_s, depth, intensity = 1.5, 4.0, 0.6, 4000.0
    light_z, top_z = 15.0, 1.8

    def build(scene):
        scene.set_background(BLACK)
        Scene.clear_lights()
        PointLight(
            location=OUT * light_z, color=WHITE, intensity=intensity, decay=2
        ).spawn(animate=False)
        floor = Prism(width=8, height=8, depth=0.1)
        floor.set_material(MeshLambertMaterial(color=BLACK))
        floor.spawn(animate=False)
        block = Prism(width=4, height=4, depth=depth)
        block.set_material(
            MeshPhysicalMaterial(
                color=WHITE, transmission=1, ior=n, roughness=0, sigma_s=sigma_s
            )
        )
        block.move(OUT * (top_z - depth / 2))
        block.spawn(animate=False)

    image, result = _render_scene_result(
        tmp_path,
        "jade.png",
        build,
        128,
        video=_small(),
        max_bounces=2,
        shadows=True,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
            "pt_caustics": caustics,
        },
    )
    top = _center_patch_mean(image, half=4)
    h, steps = light_z - top_z, 2000
    dz = depth / steps
    spread = dz * sum(
        math.exp(-2 * sigma_s * z) / (h + z / n) ** 2
        for z in ((i + 0.5) * dz for i in range(steps))
    )
    transmittance = 1 - ((n - 1) / (n + 1)) ** 2
    single = 255 * transmittance**2 / n**2 * sigma_s / (4 * math.pi) * intensity
    single *= spread
    if caustics:
        # What this pins: a scatter point used to be moved onto an unrelated
        # plane (its medium triangle read as a circuit), and a walk from one
        # just under the face could not converge in f32 (``MNEE_ROUNDING``),
        # dropping the least attenuated light. Here they cost about 2% and 3%.
        assert top == pytest.approx(single, rel=0.02)
        _assert_no_caustic_truncation(result)
    else:
        assert top < 0.02 * single, top


def _small():
    from test_path_tracer import SMALL_SETTINGS

    return SMALL_SETTINGS
