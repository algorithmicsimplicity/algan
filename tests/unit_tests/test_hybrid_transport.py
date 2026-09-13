"""Shared transport arithmetic and deterministic closed-shell continuations.

Feature tests, deliberately outside --fast. Kernel probes retain runtime
annotations; the rendered checks exercise both deterministic front ends.
"""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from algan import (
    BLACK,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Off,
    Prism,
    Scene,
    SceneManager,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shadow_dispatch import (
    _provably_opaque_shadow_batch,
    _select_shadow_mode,
)
from algan.rendering.raytracing.shell_alpha import _build_closed_shell_table
from algan.rendering.raytracing.transport_taichi import (
    _offset_ray_origin,
    _shadow_tmax,
    _transmission_normal,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _SCA_IOR_BASE,
    _SCA_IOR_DEPTH,
    _SCA_SHELL_BASE,
    SCA_WIDTH_NESTED,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _relative_ior,
    _reset_shell_segment,
    _shell_exit,
    _write_ior_stack,
    sca_width,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils.memory_utils import ManualMemory


def _array(values, dtype=ti.f32):
    values = np.asarray(values, dtype=np.float32 if dtype == ti.f32 else np.int32)
    out = ti.ndarray(dtype, shape=values.shape)
    out.from_numpy(values)
    return out


@ti.kernel
def _interface_probe(
    state: ti.types.ndarray(),
    inputs: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(inputs.shape[0]):
        rd = ti.math.vec3(inputs[i, 0], 0.0, inputs[i, 1])
        face = ti.math.vec3(0.0, 0.0, 1.0)
        shade = face * inputs[i, 3]
        n = _transmission_normal(rd, shade, face)
        rel = _relative_ior(state, i, inputs[i, 2], rd.dot(face) < 0.0, 1)
        refl, passed = _material_reflectance(
            rd, n, 0.0, inputs[i, 2], ti.math.vec3(1.0), 1.0, rel
        )
        direction = _refract_ray(rd, n, rel)
        out[i, 0] = rel
        out[i, 1] = refl[0]
        out[i, 2] = passed
        for k in ti.static(range(3)):
            out[i, 3 + k] = direction[k]


def _interface(angle, inside, material=1.5, outside=1.33, shade_sign=1):
    init_taichi()
    state = np.zeros((1, SCA_WIDTH_NESTED), np.float32)
    stack = [outside, material] if inside else [outside]
    state[0, _SCA_IOR_DEPTH] = len(stack)
    state[0, _SCA_IOR_BASE : _SCA_IOR_BASE + len(stack)] = stack
    a = math.radians(angle)
    inputs = [[math.sin(a), math.cos(a) * (1 if inside else -1), material, shade_sign]]
    out = ti.ndarray(ti.f32, shape=(1, 6))
    _interface_probe(_array(state), _array(inputs), out)
    return out.to_numpy()[0]


@pytest.mark.parametrize("inside", [False, True])
def test_interface_f0_uses_the_enclosing_medium(inside):
    rel, reflected, passed, *_ = _interface(0, inside)
    expected = ((1.5 - 1.33) / (1.5 + 1.33)) ** 2
    assert rel == pytest.approx(1.5 / 1.33, rel=1e-6)
    assert reflected == pytest.approx(expected, rel=1e-5)
    assert reflected + passed == pytest.approx(1.0)


def test_glass_in_water_does_not_inherit_glass_air_tir():
    water = _interface(50, True)
    air = _interface(50, True, outside=1.0)
    assert water[2] > 0.9
    assert water[5] > 0.0
    assert water[3] == pytest.approx(1.5 / 1.33 * math.sin(math.radians(50)), abs=1e-6)
    assert air[1] == pytest.approx(1)
    assert air[2] == pytest.approx(0)
    assert air[5] < 0.0  # Snell's TIR fallback agrees with the energy decision.


@pytest.mark.parametrize(
    ("material", "outside", "inside"), [(1.0, 1.5, False), (1.5, 1.0, True)]
)
def test_tir_on_either_side_of_a_relative_interface(material, outside, inside):
    got = _interface(50, inside, material, outside)
    assert got[1] == pytest.approx(1)
    assert got[2] == pytest.approx(0)
    assert got[5] * (1 if inside else -1) < 0


@pytest.mark.parametrize("inside", [False, True])
def test_index_matched_interfaces_transmit_without_bending(inside):
    got = _interface(62, inside, material=1.5, outside=1.5)
    assert got[1] == pytest.approx(0)
    assert got[2] == pytest.approx(1)
    assert got[3] == pytest.approx(math.sin(math.radians(62)), abs=1e-6)
    assert got[5] == pytest.approx(
        math.cos(math.radians(62)) * (1 if inside else -1), abs=1e-6
    )


@pytest.mark.parametrize("inside", [False, True])
def test_normal_mapping_cannot_reverse_the_interface_side(inside):
    ordinary = _interface(50, inside)
    flipped = _interface(50, inside, shade_sign=-1)
    np.testing.assert_allclose(flipped, ordinary, atol=1e-6)


@ti.kernel
def _offset_probe(
    points: ti.types.ndarray(), normals: ti.types.ndarray(), out: ti.types.ndarray()
):
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        n = ti.math.vec3(normals[i, 0], normals[i, 1], normals[i, 2])
        shifted = _offset_ray_origin(p, n)
        transmitted = _offset_transmitted_origin(p, n, n, -n)
        for k in ti.static(range(3)):
            out[i, k] = shifted[k]
            out[i, 3 + k] = transmitted[k]
        out[i, 6] = _shadow_tmax(p, n, 1.0)


def test_shared_offsets_cover_signed_large_coordinates_and_thin_gaps():
    init_taichi()
    points = np.array([[0, 0, v] for v in [0, 1, -1, 1e5, -1e5]], np.float32)
    normals = np.tile([0, 0, 1], (len(points), 1)).astype(np.float32)
    out = ti.ndarray(ti.f32, shape=(len(points), 7))
    _offset_probe(_array(points), _array(normals), out)
    got = out.to_numpy()
    delta = got[:, 2] - points[:, 2]
    assert (delta > 0).all()
    assert (got[:, 5] > got[:, 2]).all()  # geometric side + forward guard
    assert delta[0] < 1e-4  # does not cross a 1e-4 gap near the origin
    assert delta[3] > 1e-3  # former fixed offset rounded away
    assert delta[4] > 1e-3
    assert 0 < got[0, 6] < 1  # exclude the emitter, not a nearer blocker
    assert got[0, 6] > 0.999
    from algan.rendering.raytracing.path_tracer_taichi import (
        _pt_offset_ray_origin,
        _pt_shadow_tmax,
    )

    assert _pt_offset_ray_origin is _offset_ray_origin
    assert _pt_shadow_tmax is _shadow_tmax


@ti.kernel
def _shell_stream(
    state: ti.types.ndarray(), ids: ti.types.ndarray(), out: ti.types.ndarray()
):
    # A serialized stream, just like the hit drain of one ray.
    for row in range(1):
        for i in range(ids.shape[0]):
            out[i] = ti.cast(_shell_exit(state, row, ids[i]), ti.i32)


@ti.kernel
def _reset_and_spawn(state: ti.types.ndarray()):
    for r in range(1):
        _write_ior_stack(state, r, 1, 1.2, True, True, 1)
        _reset_shell_segment(state, 1)
        _reset_shell_segment(state, 0)


@pytest.mark.parametrize("count", [1, 24, 25, 73, 257])
def test_shell_pairing_has_no_fixed_nesting_cap_and_reentry_counts_again(count):
    init_taichi()
    state = _array(np.zeros((2, sca_width(False, count)), np.float32))
    ids = list(range(count)) + list(reversed(range(count))) + [0, 0, -1]
    out = ti.ndarray(ti.i32, shape=(len(ids),))
    _shell_stream(state, _array(ids, ti.i32), out)
    assert out.to_numpy().tolist() == [0] * count + [1] * count + [0, 1, 0]
    assert not state.to_numpy().any()


def test_full_24_bit_words_survive_and_scatter_clears_only_shell_state():
    init_taichi()
    width = sca_width(True, 48)
    source = np.zeros((2, width), np.float32)
    source[:, :_SCA_SHELL_BASE] = 3.0
    source[0, _SCA_IOR_DEPTH] = 1
    source[0, _SCA_IOR_BASE] = 1.5
    state = _array(source)
    out = ti.ndarray(ti.i32, shape=(48,))
    _shell_stream(state, _array(range(48), ti.i32), out)
    assert (state.to_numpy()[0, _SCA_SHELL_BASE:] == 2**24 - 1).all()
    _reset_and_spawn(state)
    got = state.to_numpy()
    np.testing.assert_array_equal(got[0, :_SCA_SHELL_BASE], source[0, :_SCA_SHELL_BASE])
    assert got[1, _SCA_IOR_DEPTH] == 2
    assert got[1, _SCA_IOR_BASE + 1] == pytest.approx(1.2)
    assert not got[:, _SCA_SHELL_BASE:].any()


def test_shell_table_is_dense_broadcast_and_arena_backed(monkeypatch):
    monkeypatch.setattr(rt_settings, "solid_shell_alpha", True)
    memory = ManualMemory(0, device="cpu", num_bytes=4096)
    merged = {
        "tri_obj": torch.tensor([[123, 123, 999999, 999999, -1]], dtype=torch.int32),
        "tri_closed": torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32
        ),
        "tri_has_translucent": True,
    }
    table, count = _build_closed_shell_table(memory, merged)
    assert count == 2
    assert table.tolist() == [[0, 0, -1, -1, -1], [0, 0, 1, 1, -1]]
    assert (
        table.untyped_storage().data_ptr() == memory.data.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("disable", ["setting", "opaque", "no_closed"])
def test_unused_shell_tables_allocate_only_the_disabled_stub(monkeypatch, disable):
    monkeypatch.setattr(rt_settings, "solid_shell_alpha", disable != "setting")
    merged = {
        "tri_obj": torch.tensor([[3, 3]]),
        "tri_closed": torch.tensor([[1.0, 1.0]]),
        "tri_has_translucent": True,
    }
    if disable == "opaque":
        merged["tri_has_translucent"] = False
    elif disable == "no_closed":
        merged["tri_closed"].zero_()
    table, count = _build_closed_shell_table(
        ManualMemory(0, device="cpu", num_bytes=64), merged
    )
    assert count == 0
    assert table.tolist() == [[-1]]
    assert sca_width(False, count) == 7
    assert sca_width(True, count) == 12


@pytest.mark.parametrize("count", [1, 24, 25, 1000])
def test_tile_planner_charges_every_shell_word(count):
    from algan.rendering.raytracing.tracer import _wavefront_state_coefficients

    width = sca_width(False, count)
    assert width == 12 + (count + 23) // 24
    plain = _wavefront_state_coefficients(7)
    expanded = _wavefront_state_coefficients(width)
    assert expanded["pool"] - plain["pool"] == (width - 7) * 4
    assert expanded["primary"] == plain["primary"]
    assert expanded["fixed"] == plain["fixed"]


_OPACITY_FACTS = (
    "has_transmissive",
    "tri_has_translucent",
    "bez_has_translucent",
    "has_uncertain_texture_alpha",
)


@pytest.mark.parametrize("fact", _OPACITY_FACTS)
@pytest.mark.parametrize("missing", [False, True])
def test_auto_shadows_require_all_four_opacity_facts(fact, missing):
    merged = dict.fromkeys(_OPACITY_FACTS, False)
    assert _provably_opaque_shadow_batch(merged)
    assert _select_shadow_mode(True, "auto", merged) == 3
    if missing:
        del merged[fact]
    else:
        merged[fact] = True
    assert not _provably_opaque_shadow_batch(merged)
    assert _select_shadow_mode(True, "auto", merged) == 1


def test_shadow_switches_preserve_reference_and_explicit_experimental_modes():
    merged = dict.fromkeys(_OPACITY_FACTS, False)
    assert _select_shadow_mode(False, "auto", merged) == 0
    assert _select_shadow_mode(True, False, merged) == 1
    assert _select_shadow_mode(True, "gather", merged) == 4
    merged["tri_has_translucent"] = True
    assert _select_shadow_mode(True, True, merged) == 2
    merged["has_transmissive"] = True
    assert _select_shadow_mode(True, True, merged) == 1
    SETTINGS.raytracing.experimental.set(shadow_anyhit="auto")
    assert rt_settings.shadow_anyhit == "auto"
    SETTINGS.raytracing.experimental.set(shadow_anyhit=False)
    assert rt_settings.shadow_anyhit is False
    SETTINGS.raytracing.experimental.set(shadow_anyhit="auto")
    assert rt_settings.shadow_anyhit == "auto"


def _render_mirror_shell(tmp_path, name, opacity):
    video = SMOKE_TEST.set(resolution=(32, 32))
    SceneManager.reset()
    try:
        with Scene(video_settings=video) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                mirror = Prism(width=3, height=3, depth=0.2)
                mirror.set_material(
                    MeshStandardMaterial(color=WHITE, metalness=1, roughness=0)
                )
                mirror.rotate(45, UP).spawn(animate=False)
                for side in (RIGHT, -RIGHT):
                    shell = Prism(width=0.4, height=8, depth=6, opacity=opacity)
                    shell.set_material(
                        MeshBasicMaterial(color=(0.5, 0, 0), opacity=opacity)
                    )
                    shell.move(side * 5).spawn(animate=False)
            result = scene.save_frame(
                tmp_path / name, video_settings=video, overwrite=True
            )
        assert result.render_plan.truncations.dropped_continuations == 0
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32)[
                14:18, 14:18, 0
            ].mean()
    finally:
        SceneManager.reset()


@pytest.mark.parametrize("hybrid", [True, False], ids=["sheets", "classic"])
def test_deterministic_mirror_preserves_authored_shell_opacity(
    tmp_path, monkeypatch, hybrid
):
    from algan.rendering.raytracing import raster_pipeline

    calls = []
    original = raster_pipeline.shade_sparse_raster_coverage

    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(raster_pipeline, "shade_sparse_raster_coverage", recording)
    SETTINGS.raytracing.set(
        samples_per_pixel=1,
        shadows=False,
        linear_color_space=False,
        tonemapping=False,
        max_bounces=3,
    )
    SETTINGS.raytracing.experimental.set(
        hybrid_raster=hybrid,
        post_process_tonemap=False,
        solid_shell_alpha=True,
        analytic_aa_secondary_samples=1,
    )
    opaque = _render_mirror_shell(tmp_path, "opaque.png", 1)
    half = _render_mirror_shell(tmp_path, "half.png", 0.5)
    assert opaque > 30, "the mirror must see the shell"
    assert half == pytest.approx(0.5 * opaque, abs=1.5)
    assert bool(calls) == hybrid, "the named front end was not exercised"
    # Prove the assertion detects the old double-shell composite.
    SETTINGS.raytracing.experimental.set(solid_shell_alpha=False)
    doubled = _render_mirror_shell(tmp_path, "per_crossing.png", 0.5)
    assert doubled == pytest.approx(0.75 * opaque, abs=1.5)


@pytest.mark.parametrize("hybrid", [True, False], ids=["sheets", "classic"])
def test_nested_interface_weights_reach_the_rendered_continuations(tmp_path, hybrid):
    """A normal-incidence slab has an independently calculable transmitted share.

    Four crossings: air/water, water/glass, glass/water, water/air. The direct
    term is (1-F_air_water)^2 (1-F_water_glass)^2; extra internal round trips
    add less than a byte here. Treating glass as surrounded by air loses ~7%.
    """
    from algan import INWARD, MeshPhysicalMaterial

    video = SMOKE_TEST.set(resolution=(32, 32))
    SETTINGS.raytracing.set(
        samples_per_pixel=1,
        shadows=False,
        linear_color_space=False,
        tonemapping=False,
        max_bounces=8,
    )
    SETTINGS.raytracing.experimental.set(
        hybrid_raster=hybrid,
        nested_ior=True,
        post_process_tonemap=False,
        analytic_aa_secondary_samples=1,
    )
    SceneManager.reset()
    try:
        with Scene(video_settings=video) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                for width, depth, ior in [(3, 2, 1.33), (2, 1, 1.5)]:
                    glass = Prism(width=width, height=width, depth=depth)
                    glass.set_material(
                        MeshPhysicalMaterial(
                            color=WHITE, roughness=0, transmission=1, ior=ior
                        )
                    )
                    glass.spawn(animate=False)
                back = Prism(width=12, height=12, depth=0.1)
                back.set_material(MeshBasicMaterial(color=WHITE))
                back.move(INWARD * 3).spawn(animate=False)
            result = scene.save_frame(
                tmp_path / "nested.png", video_settings=video, overwrite=True
            )
        with Image.open(result.output_path) as image:
            measured = (
                np.asarray(image.convert("RGB"), dtype=np.float32)[15:17, 15:17].mean()
                / 255
            )
        f_outer = ((1.33 - 1) / (1.33 + 1)) ** 2
        f_inner = ((1.5 - 1.33) / (1.5 + 1.33)) ** 2
        direct = (1 - f_outer) ** 2 * (1 - f_inner) ** 2
        assert measured == pytest.approx(direct, abs=2 / 255)
        assert result.render_plan.truncations.dropped_continuations == 0
    finally:
        SceneManager.reset()


def test_classic_shell_state_survives_multiple_event_batches(tmp_path):
    """Six nested shells exceed a four-slot ring and the usual hit-buffer width."""
    from algan.rendering.raytracing.raytrace_kernels_taichi import kbuf

    count = max(6, int(kbuf) + 1)
    opacity = 0.08
    SETTINGS.raytracing.set(
        samples_per_pixel=1, shadows=False, linear_color_space=False, tonemapping=False
    )
    SETTINGS.raytracing.experimental.set(
        hybrid_raster=False, solid_shell_alpha=True, post_process_tonemap=False
    )
    video = SMOKE_TEST.set(resolution=(32, 32))
    SceneManager.reset()
    try:
        with Scene(video_settings=video) as scene:
            with Off():
                scene.set_background(BLACK)
                Scene.clear_lights()
                for i in range(count):
                    shell = Prism(
                        width=3, height=3, depth=0.4 + i * 0.12, opacity=opacity
                    )
                    shell.set_material(MeshBasicMaterial(color=WHITE, opacity=opacity))
                    shell.spawn(animate=False)
            result = scene.save_frame(
                tmp_path / "nested_shells.png", video_settings=video, overwrite=True
            )
        with Image.open(result.output_path) as image:
            measured = (
                np.asarray(image.convert("RGB"), dtype=np.float32)[14:18, 14:18].mean()
                / 255
            )
        assert measured == pytest.approx(1 - (1 - opacity) ** count, abs=1.5 / 255)
        assert result.render_plan.truncations.surfaces_per_ray == 0
    finally:
        SceneManager.reset()
