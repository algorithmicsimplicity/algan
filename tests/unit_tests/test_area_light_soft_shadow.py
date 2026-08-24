"""A rect area light's packed rows carry their emitter CELL's geometry.

Each ``RectAreaLight`` row used to leave the shadow-radius column at zero, so
every row took the single-hard-ray path and the union of K hard shadows was a
staircase. The fix packs, per row: the cell's half-extents along the
rectangle's ``right``/``up`` (aux 6/7), the cell's equal-area disk radius
(aux 8 -- the fans' gate and isotropic fallback), and the rectangle's ``right``
unit axis (aux 9-11; ``up`` is recovered in-kernel as ``cross(normal,
right)``).

Most of this file holds pure host-side assertions on the packed aux: no
render, no kernel. They pin the external contract the kernel fans rely on
(unit ``right``, orthogonal to the packed normal, cells tiling the rectangle
exactly) and the regression guarantee that no other light type's columns
moved. The render tests at the bottom are the exception: they exist to
compile the two shadow fans, which no host-side assertion can reach.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan import (
    BLACK,
    ORIGIN,
    OUT,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    Off,
    Prism,
    Scene,
    Square,
)
from algan.rendering.lights import (
    DirectionalLight,
    HemisphereLight,
    PointLight,
    RectAreaLight,
    SpotLight,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.utils.color_space import srgb_to_linear


def _light(**kwargs):
    return RectAreaLight(target=(0.0, 0.0, 0.0), **kwargs)


def _aux(light, loc=(0.0, 0.0, 3.0)):
    """Packed aux for one frame, ``[K, 13]``, with the light facing -z."""
    location = torch.tensor([loc], dtype=torch.float32)
    return light.build_aux(location)[0]


def test_cells_pack_half_extents_and_equal_area_radius():
    with Scene():
        aux = _aux(_light(width=1.8, height=1.0, samples=4))
    hu, hv = 1.8 / 4, 1.0 / 4
    assert aux[0, 6].item() == pytest.approx(hu)
    assert aux[0, 7].item() == pytest.approx(hv)
    # Every row stands for an equally sized cell: the extents are uniform.
    assert torch.allclose(aux[:, 6], torch.full((4,), hu))
    assert torch.allclose(aux[:, 7], torch.full((4,), hv))
    assert aux[0, 8].item() == pytest.approx(math.sqrt(4.0 * hu * hv / math.pi))


def test_packed_right_axis_is_unit_and_orthogonal_to_the_normal():
    """The external invariant the kernel's ``cross(normal, right)`` depends
    on: without it the recovered ``up`` axis would not span the light plane.
    """
    with Scene():
        light = _light(width=1.8, height=1.0, samples=4)
        location = torch.tensor([(0.0, 0.0, 3.0)])
        aux = light.build_aux(location)[0]
        right, _up = light._rect_axes(location)
    normal = aux[0, 3:6]
    packed_right = aux[0, 9:12]
    assert packed_right.norm().item() == pytest.approx(1.0, abs=1e-6)
    assert packed_right.dot(normal).item() == pytest.approx(0.0, abs=1e-6)
    # The packed axis is exactly the geometry the light itself lays out.
    assert torch.allclose(packed_right, right[0], atol=1e-6)


def test_cells_tile_the_rectangle():
    """Sample positions +/- (hu, hv) along (right, up) cover the rectangle
    exactly -- what makes the per-cell integral sum to the whole-rectangle
    integral instead of double-covering or missing strips.
    """
    with Scene():
        light = _light(width=1.8, height=1.0, samples=4)
        location = torch.tensor([(0.2, -0.4, 3.0)])
        aux = light.build_aux(location)[0]
        k = light._grid_side()
        hu, hv = aux[0, 6].item(), aux[0, 7].item()
        assert 2 * k * hu == pytest.approx(1.8)
        assert 2 * k * hv == pytest.approx(1.0)

        right, up = light._rect_axes(location)
        positions = light.get_sample_positions(location)[0]  # [K, 3]
        rel = positions - location[0]
        along_r = rel @ right[0]
        along_u = rel @ up[0]
    assert (along_r.max() + hu).item() == pytest.approx(1.8 / 2)
    assert (along_r.min() - hu).item() == pytest.approx(-1.8 / 2)
    assert (along_u.max() + hv).item() == pytest.approx(1.0 / 2)
    assert (along_u.min() - hv).item() == pytest.approx(-1.0 / 2)


def test_single_sample_cell_is_the_whole_rectangle():
    with Scene():
        aux = _aux(_light(width=1.8, height=1.0, samples=1))
    assert aux.shape[0] == 1
    assert aux[0, 6].item() == pytest.approx(1.8 / 2)
    assert aux[0, 7].item() == pytest.approx(1.0 / 2)


def test_non_square_rectangle_keeps_its_aspect():
    """An 18:1 emitter keeps elongated cells; nothing squares them off into
    equal-area disks.
    """
    with Scene():
        aux = _aux(_light(width=1.8, height=0.1, samples=4))
    hu, hv = aux[0, 6].item(), aux[0, 7].item()
    assert hu != hv
    assert hu == pytest.approx(1.8 / 4)
    assert hv == pytest.approx(0.1 / 4)


def test_other_light_types_pack_aux_6_to_11_exactly_as_before():
    """Regression guard for the kernels' ``ltype`` guard: columns 9/10 are a
    spot light's cone cosines, column 8 every soft emitter's radius, and
    nothing outside the area type may grow data there.
    """

    def _row(light, loc=(0.0, 0.0, 0.0)):
        return light.build_aux(torch.tensor([loc], dtype=torch.float32))[0, 0]

    with Scene():
        point = _row(PointLight(decay=2, distance=5.0, shadow_radius=0.25))
        spot = _row(
            SpotLight(angle=30.0, penumbra=0.5, shadow_radius=0.1),
            loc=(0.0, 0.0, 3.0),
        )
        directional = _row(DirectionalLight(shadow_angle=10.0), loc=(0.0, 0.0, 3.0))
        hemisphere = _row(HemisphereLight(ground_color=(0.25, 0.5, 1.0)))

    assert point[1].item() == 2.0
    assert point[2].item() == 5.0
    assert point[8].item() == pytest.approx(0.25)
    assert not point[6:8].any()
    assert not point[9:12].any()

    outer = math.cos(math.radians(30.0))
    inner = math.cos(math.radians(30.0) * (1.0 - 0.5))
    assert spot[6].item() == pytest.approx(outer)
    assert spot[7].item() == pytest.approx(inner)
    assert spot[8].item() == pytest.approx(0.1)
    assert not spot[9:12].any()

    assert directional[8].item() == pytest.approx(math.tan(math.radians(10.0) * 0.5))
    assert not directional[6:8].any()
    assert not directional[9:12].any()

    ground = torch.tensor((0.25, 0.5, 1.0))
    if rt_settings.LINEAR_COLOR_SPACE:
        ground = srgb_to_linear(ground)
    assert not hemisphere[6:9].any()
    assert torch.allclose(hemisphere[9:12], ground)


def test_flag_off_packs_todays_row(monkeypatch):
    """With the toggle off, build_aux writes zeros to aux 6-11: bit-for-bit
    today's row, so the kernels take their existing single-ray path.
    """
    monkeypatch.setattr(rt_settings, "AREA_LIGHT_SOFT_SHADOWS", False)
    with Scene():
        aux = _aux(_light(width=1.8, height=1.0, samples=4))
    assert not aux[:, 6:12].any()
    # Everything the base packing always carried is untouched.
    assert aux[0, 0].item() == pytest.approx(5.0)
    assert aux[0, 12].item() == pytest.approx(0.25)
    assert torch.allclose(aux[0, 3:6], torch.tensor((0.0, 0.0, -1.0)))


def test_experimental_setting_surfaces_and_drives_the_legacy_global():
    """``SETTINGS.raytracing.experimental.area_light_soft_shadows`` is the
    supported way to flip the flag, and it writes the global build_aux reads.
    """
    previous = SETTINGS.raytracing.experimental.area_light_soft_shadows
    try:
        SETTINGS.raytracing.experimental.area_light_soft_shadows = False
        assert rt_settings.AREA_LIGHT_SOFT_SHADOWS is False
        with Scene():
            aux = _aux(_light(samples=4))
        assert not aux[:, 6:12].any()

        SETTINGS.raytracing.experimental.area_light_soft_shadows = True
        assert rt_settings.AREA_LIGHT_SOFT_SHADOWS is True
        with Scene():
            aux = _aux(_light(samples=4))
        assert aux[:, 6:12].any()
    finally:
        SETTINGS.raytracing.experimental.area_light_soft_shadows = previous


# --------------------------------------------------------------------------
# Render tests. A host-side assertion cannot catch a kernel that does not
# COMPILE -- that is exactly how the wavefront fan's out-of-scope ``off``
# defect shipped -- so these exist to compile the two shadow fans for real,
# one frame each at 32x32, and assert nothing about pixels.
# --------------------------------------------------------------------------


def _render_one_area_shadow_frame(tmp_path, name):
    """Render one 32x32 frame of an area light over a blocked ground plane.

    Kept deliberately minimal: the assertion downstream is "the kernel
    compiled and the frame rendered", so the scene only needs to route shadow
    rays through a soft area emitter.
    """
    output_path = tmp_path / name
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            scene.set_background_color(BLACK)
            with Off():
                Scene.clear_light_sources()
                RectAreaLight(
                    location=UP * 5.0,
                    target=ORIGIN,
                    width=4.0,
                    height=4.0,
                    samples=4,
                    color=WHITE,
                    intensity=1.2,
                ).spawn(animate=False)
                (
                    Prism(dimensions=(6.0, 6.0, 0.1))
                    .set_material(MeshLambertMaterial(color=WHITE))
                    .spawn(animate=False)
                )
                (
                    Square(side_length=1.2, color=WHITE)
                    .move(RIGHT * 0.8 + OUT * 1.5)
                    .spawn(animate=False)
                )
            result = scene.save_frame(
                output_path,
                video_settings=SMOKE_TEST,
                overwrite=True,
            )
    finally:
        SceneManager.reset()
    return result


@pytest.mark.parametrize("analytic_aa", [False, True], ids=("wavefront", "sheet"))
def test_soft_shadow_fans_compile_and_render_one_frame(
    tmp_path,
    monkeypatch,
    analytic_aa,
):
    """Both deterministic shadow fans COMPILE with a rect area emitter present.

    This is not a redundant pixel test and must stay unmarked (never ``fast``):
    its whole purpose is to compile kernels, which no host-side unit test in
    this file can do. Taichi rejects an out-of-scope local at kernel-compile
    time -- ``wavefront_shade``'s soft-shadow fan once shipped assigning
    ``off`` inside both arms of an if/else and reading it after -- and nothing
    else in the suite routes a soft area light through the classic wavefront
    shade kernel, so deleting this test re-opens that blind spot.

    Each arm forces one fan:

    * ``analytic_aa=False`` vetoes the sheet route in
      ``analytic_raster_route_active`` (the single host-side route decision),
      so the batch falls back to the classic wavefront tracer and compiles
      the inline fan in ``wavefront_shade``;
    * the default sheet arm compiles ``raster_shadow_trace``'s fan via the
      sheet resolve's mode-1 event build.

    The spy on ``analytic_raster_route_active`` pins which decision was
    actually made, so a future routing change cannot silently empty an arm
    of its purpose while staying green.
    """
    expected_route_active = analytic_aa
    decisions = []
    real_decision = tracer.analytic_raster_route_active

    def _spy(*args, **kwargs):
        active = real_decision(*args, **kwargs)
        decisions.append(active)
        return active

    monkeypatch.setattr(tracer, "analytic_raster_route_active", _spy)

    # Shadows on engages both fans' soft-emitter path (an area row carries a
    # non-zero radius); analytic_aa selects which fan resolves the frame.
    SETTINGS.raytracing.set(shadows=True, analytic_aa=analytic_aa)

    result = _render_one_area_shadow_frame(tmp_path, f"area_soft_{analytic_aa}")

    assert decisions, "the render never consulted the route decision"
    assert set(decisions) == {expected_route_active}, (
        f"expected the {'sheet' if expected_route_active else 'classic wavefront'} "
        f"route, route decision reported {decisions}"
    )
    assert result.output_path.exists()
