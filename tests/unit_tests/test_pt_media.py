"""Homogeneous volumes and shell-scoped random-walk subsurface scattering."""

from __future__ import annotations

import math

import pytest
import torch

from algan import (
    BLACK,
    ORIGIN,
    OUT,
    SMOKE_TEST,
    WHITE,
    DirectionalLight,
    MeshPhysicalMaterial,
    PointLight,
    Prism,
    Scene,
)
from algan.rendering.taichi_runtime import init_taichi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _samples(n=32768):
    return torch.quasirandom.SobolEngine(2, scramble=True, seed=413).draw(n).to(DEVICE)


@pytest.mark.parametrize("sigma", [-1, float("nan"), float("inf"), (0, 1, -1), (1, 2)])
def test_invalid_scattering_coefficient(sigma):
    with pytest.raises(ValueError, match="sigma_s"):
        MeshPhysicalMaterial(sigma_s=sigma)


@pytest.mark.parametrize("g", [-1, 1, 2, float("nan"), float("inf")])
def test_invalid_phase_anisotropy(g):
    with pytest.raises(ValueError, match="g must"):
        MeshPhysicalMaterial(g=g)


def test_scattering_is_numeric_not_srgb():
    material = MeshPhysicalMaterial(sigma_s=(0.1, 2, 30), g=0.6)
    values = material.get_shader_param_values()
    assert torch.allclose(values["sigma_s"], torch.tensor([0.1, 2, 30]))
    assert values["g"] == 0.6
    assert torch.equal(MeshPhysicalMaterial().sigma_s, torch.zeros(3))


@pytest.mark.parametrize("g", [-0.8, -0.2, -1e-4, 0.0, 1e-4, 0.3, 0.8])
def test_hg_normalization_mean_and_matched_pdf(g):
    from pt_media_probe_taichi import phase_probe

    init_taichi()
    samples = _samples()
    out = torch.zeros((len(samples), 4), device=DEVICE)
    phase_probe(g, samples, out)
    assert float(out[:, 0].mean()) == pytest.approx(g, abs=0.002)
    assert torch.allclose(out[:, 1], torch.ones_like(out[:, 1]), atol=2e-6)
    assert torch.allclose(out[:, 2], out[:, 3], rtol=2e-5, atol=1e-6)
    # E_{p}[1 / p] is the sphere's area, not a hemisphere's area.
    assert float((1 / out[:, 2]).mean()) == pytest.approx(4 * torch.pi, rel=0.005)


@pytest.mark.parametrize(
    ("sa", "ss"), [((0.2, 0.7, 1.1), (0.5, 2, 3)), ((0, 0, 0), (0, 1, 2))]
)
def test_rgb_flight_slab_closed_form(sa, ss):
    from pt_media_probe_taichi import flight_probe

    init_taichi()
    samples = _samples(65536)
    out = torch.zeros((len(samples), 5), device=DEVICE)
    a = torch.tensor(sa, device=DEVICE)
    s = torch.tensor(ss, dtype=torch.float32, device=DEVICE)
    a = a.float()
    length = 0.7
    flight_probe(a, s, length, samples, out)
    st = a + s
    trans = torch.exp(-st * length)
    collision = out[:, 0:1]
    escape_estimate = ((1 - collision) * out[:, 2:5]).mean(0)
    scatter_estimate = (collision * out[:, 2:5]).mean(0)
    reference_scatter = s / st.clamp_min(1e-30) * (1 - trans)
    assert torch.allclose(escape_estimate, trans, atol=0.003), (escape_estimate, trans)
    assert torch.allclose(scatter_estimate, reference_scatter, atol=0.003)


def test_stack_tracks_object_not_triangle_and_reports_overflow():
    from pt_media_probe_taichi import stack_probe

    init_taichi()
    shell = torch.tensor(
        [[-1] * 6 + [10, 10, 20, 30, 40, 50]], dtype=torch.int32, device=DEVICE
    )
    events = torch.tensor(
        [[0, 1], [1, 1], [2, 1], [1, 0], [0, 1], [3, 1], [4, 1], [5, 1]],
        dtype=torch.int32,
        device=DEVICE,
    )
    out = torch.empty((len(events), 5), dtype=torch.int32, device=DEVICE)
    stack_probe(shell, events, out)
    result = out.cpu().tolist()
    assert result[1] == [0, -1, -1, -1, 0], "same shell edge must not enter twice"
    assert result[3] == [2, -1, -1, -1, 0], "exit must preserve the other shell"
    assert result[-1][-1] == 1


def _fog_scene(sigma, g=0.0, interior=False):
    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        fog = Prism(width=3, height=3, depth=2)
        fog.set_material(
            MeshPhysicalMaterial(
                color=WHITE, transmission=1, ior=1, roughness=0, sigma_s=sigma, g=g
            )
        )
        fog.spawn(animate=False)
        PointLight(location=OUT * 3, color=WHITE, intensity=1, decay=0).spawn(
            animate=False
        )
        if interior:
            scene.camera.move_to(ORIGIN)
            scene.camera.look_at(-OUT)

    return build


@pytest.mark.parametrize("interior", [False, True])
def test_volume_scatter_renders_with_exterior_and_interior_camera(tmp_path, interior):
    from test_path_tracer import _render_scene_exp

    image = _render_scene_exp(
        tmp_path,
        f"fog_{interior}.png",
        _fog_scene(0.5, interior=interior),
        16,
        video=SMOKE_TEST,
        max_bounces=4,
        experimental={"pt_error_target": 0, "pt_firefly_clamp": 0},
    )
    assert torch.isfinite(image.float()).all()
    assert float(image[..., :3].float().mean()) > 0.5, "the volume must scatter light"


@pytest.mark.parametrize("shadows", [True, False])
def test_homogeneous_slab_single_scatter_matches_closed_form(tmp_path, shadows):
    """Full renderer reference: a collimated beam crosses a two-unit slab.

    At one allowed volume collision, back lighting gives a constant integrand
    sigma_s * phase(1) * exp(-sigma_t * D). Absorption is applied to BOTH
    camera and light segments, once each, including when surface shadows are
    disabled. A narrow field of view keeps obliquity below 0.01%.
    """
    from test_path_tracer import _render_scene_exp

    ss = 0.8
    sa = 0.2
    depth = 2.0
    radiance = 4.0

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        scene.camera.set_fov(1)
        slab = Prism(width=10, height=10, depth=depth)
        slab.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                sigma_s=ss,
                attenuation_color=(math.exp(-sa),) * 3,
                attenuation_distance=1,
            )
        )
        slab.spawn(animate=False)
        DirectionalLight(
            location=-OUT * 5, target=ORIGIN, color=WHITE, intensity=radiance
        ).spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        f"slab_{shadows}.png",
        build,
        512,
        video=SMOKE_TEST,
        max_bounces=1,
        shadows=shadows,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
        },
    )
    expected = radiance * ss * depth * math.exp(-(sa + ss) * depth) / (4 * math.pi)
    actual = float(image[..., :3].float().mean()) / 255
    assert actual == pytest.approx(expected, abs=0.004), (actual, expected)


@pytest.mark.parametrize("ior", [1.0, 1.35])
def test_dense_conservative_cube_recovers_constant_environment(tmp_path, ior):
    """The conservative diffusion limit is a uniform radiation field.

    An optically thick shell must not grow darker as its walk repeatedly
    scatters. Check it below saturation so energy gains cannot hide either.
    The max-depth setting, not an internal hidden SSS step count, caps walks.
    """
    from test_path_tracer import _render_scene_exp

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        scene.set_environment_map(torch.full((4, 8, 3), 0.25), ambient=False)
        scene.camera.set_fov(1)
        cube = Prism(width=2, height=2, depth=2)
        cube.set_material(
            MeshPhysicalMaterial(
                color=WHITE, transmission=1, ior=ior, roughness=0, sigma_s=4
            )
        )
        cube.spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        "diffusion_furnace.png",
        build,
        256,
        video=SMOKE_TEST.set(resolution=(16, 16)),
        max_bounces=512,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
        },
    )
    actual = float(image[..., :3].float().mean()) / 255
    assert actual == pytest.approx(0.25, abs=0.015), actual


@pytest.fixture
def packed_media(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from algan.rendering.raytracing import tracer
    from algan.rendering.raytracing.settings import _MAT_DEFAULTS
    from algan.rendering.raytracing.shading_taichi import _MID_PHYSICAL

    monkeypatch.setattr(tracer, "_arena_copy", lambda _memory, value: value.clone())
    memory = SimpleNamespace(scope=lambda *_args, **_kwargs: nullcontext())
    mat = torch.tensor(_MAT_DEFAULTS).reshape(1, 1, -1).repeat(1, 3, 1)
    mat[..., 38] = 1
    merged = {
        "tri_mat": mat,
        "tri_mat_id": torch.full((1, 3), _MID_PHYSICAL, dtype=torch.int32),
        "tri_obj": torch.tensor([[10, 20, 30]], dtype=torch.int32),
        "tri_extra": torch.ones((1, 3, 15)),
    }
    return memory, merged, torch.full((1, 1), -1, dtype=torch.int32)


def test_no_scattering_keeps_original_scene_arrays(packed_media):
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory, merged, opacity = packed_media
    shells, extra, offset = _prepare_media(memory, merged, opacity)
    assert shells is opacity
    assert extra is merged["tri_extra"]
    assert offset == 0


def test_medium_table_keeps_cavities_and_does_not_mutate_shared_absorption(
    packed_media,
):
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory, merged, opacity = packed_media
    merged["tri_mat"][0, 0, 34:37] = 1
    # Third object is not closed, but has no scattering and retains its
    # existing surface-chord treatment. Second is a zero-density cavity.
    merged["tri_mat"][0, 2, 38] = 0
    shells, extra, offset = _prepare_media(memory, merged, opacity)
    assert offset == 3
    assert shells.tolist() == [[-1, -1, -1, 10, 20, -1]]
    assert not extra[..., :2, 12:15].any()
    assert extra[..., 2, 12:15].eq(1).all()
    assert merged["tri_extra"].eq(1).all(), "shared merge must survive retries"


@pytest.mark.parametrize("animated", ["material", "identity"])
def test_medium_packing_broadcasts_independently_collapsed_time_rows(
    packed_media, animated
):
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory, merged, opacity = packed_media
    merged["tri_mat"][0, 0, 34:37] = 1
    key = "tri_mat" if animated == "material" else "tri_mat_id"
    merged[key] = merged[key].repeat(2, *([1] * (merged[key].ndim - 1)))
    shells, extra, offset = _prepare_media(memory, merged, opacity)
    assert shells.shape == (2, 6)
    assert extra.shape == (2, 3, 15)
    assert offset == 3


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_animated_invalid_scattering_is_not_silently_disabled(packed_media, invalid):
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory, merged, opacity = packed_media
    merged["tri_mat"][0, 0, 34] = invalid
    with pytest.raises(ValueError, match="sigma_s"):
        _prepare_media(memory, merged, opacity)


def test_open_scattering_geometry_is_rejected(packed_media):
    from algan.rendering.raytracing.pt_media import _prepare_media

    memory, merged, opacity = packed_media
    merged["tri_mat"][0, 0, 34:37] = 1
    merged["tri_mat"][0, 0, 38] = 0
    with pytest.raises(ValueError, match="closed_shell=True"):
        _prepare_media(memory, merged, opacity)


def test_scattering_render_plan_points_to_path_tracer():
    from algan.rendering.raytracing.tracer import _build_render_plan

    merged = {"has_scattering_media": True}
    deterministic = _build_render_plan(1, None, merged)
    assert not deterministic.is_supported
    assert "scattering" in deterministic.unsupported_features[0]
    path = _build_render_plan(2, None, merged)
    assert path.is_supported
    assert "scattering" in path.requested_features[0]


@pytest.mark.parametrize("interior", [False, True])
def test_nested_vacuum_cavity_preserves_shell_absorption_and_depth_budget(
    tmp_path, interior
):
    from test_path_tracer import _render_scene_exp

    def build(scene):
        Scene.clear_lights()
        scene.set_background(WHITE)
        scene.camera.set_fov(1)
        for depth, sigma in ((4, 0.3), (2, 0)):
            shell = Prism(width=8, height=8, depth=depth)
            shell.set_material(
                MeshPhysicalMaterial(
                    color=WHITE, transmission=1, ior=1, roughness=0, sigma_s=sigma
                )
            )
            shell.spawn(animate=False)
        if interior:
            scene.camera.move_to(ORIGIN)
            scene.camera.look_at(-OUT)

    # Zero allowed scatters: isolate ballistic transmission, including four
    # null boundary crossings, without stochastic or indirect illumination.
    image = _render_scene_exp(
        tmp_path,
        f"cavity_{interior}.png",
        build,
        2,
        video=SMOKE_TEST,
        max_bounces=0,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
        },
    )
    length = 1 if interior else 2
    assert float(image[..., :3].float().mean()) / 255 == pytest.approx(
        math.exp(-0.3 * length), abs=0.004
    )


def test_light_inside_medium_uses_partial_shadow_chord(tmp_path):
    from test_path_tracer import _render_scene_exp

    sigma = 0.6
    radiance = 2.0

    def build(scene):
        Scene.clear_lights()
        scene.set_background(BLACK)
        scene.camera.set_fov(1)
        scene.camera.move_to(ORIGIN)
        scene.camera.look_at(-OUT)
        fog = Prism(width=4, height=4, depth=2)
        fog.set_material(
            MeshPhysicalMaterial(
                color=WHITE, transmission=1, ior=1, roughness=0, sigma_s=sigma
            )
        )
        fog.spawn(animate=False)
        PointLight(location=ORIGIN, color=WHITE, intensity=radiance, decay=0).spawn(
            animate=False
        )

    image = _render_scene_exp(
        tmp_path,
        "interior_light.png",
        build,
        512,
        video=SMOKE_TEST,
        max_bounces=1,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
        },
    )
    expected = radiance * (1 - math.exp(-2 * sigma)) / (8 * math.pi)
    assert float(image[..., :3].float().mean()) / 255 == pytest.approx(
        expected, abs=0.004
    )


@pytest.mark.parametrize("sampling", ["off", "always"])
def test_authored_lighting_extinction_survives_surface_shadow_switches(
    tmp_path, sampling
):
    from test_path_tracer import _render_scene_exp

    from algan import MeshToonMaterial

    def build(sigma, intensity):
        def scene_fn(scene):
            Scene.clear_lights()
            scene.set_background(BLACK)
            scene.camera.set_fov(1)
            if sigma > 0:
                fog = Prism(width=8, height=8, depth=2)
                fog.set_material(
                    MeshPhysicalMaterial(
                        color=WHITE, transmission=1, ior=1, roughness=0, sigma_s=sigma
                    )
                )
                fog.casts_shadows = False
                fog.spawn(animate=False)
            target = Prism(width=8, height=8, depth=0.2)
            target.set_material(MeshToonMaterial(color=WHITE))
            target.receives_shadows = False
            target.move_to(-OUT * 2)
            target.spawn(animate=False)
            DirectionalLight(
                location=OUT * 5, target=ORIGIN, color=WHITE, intensity=intensity
            ).spawn(animate=False)

        return scene_fn

    # Authored shaders have a direction-less ambient fill, which is not
    # light-row radiance and only traverses the camera segment. Subtract two
    # unsaturated light intensities to isolate the direct-light term instead
    # of incorrectly requiring that fill to have a shadow-ray chord too.
    direct = []
    for sigma in (0, 0.25):
        images = []
        for intensity in (0.25, 0.5):
            images.append(
                _render_scene_exp(
                    tmp_path,
                    f"authored_{sampling}_{sigma}_{intensity}.png",
                    build(sigma, intensity),
                    2,
                    video=SMOKE_TEST,
                    max_bounces=0,
                    shadows=False,
                    linear_color_space=False,
                    tonemapping=False,
                    experimental={
                        "post_process_tonemap": False,
                        "pt_error_target": 0,
                        "pt_firefly_clamp": 0,
                        "pt_authored_light_sampling": sampling,
                    },
                )
            )
        low, high = [float(image[..., :3].float().mean()) for image in images]
        direct.append(high - low)
    clear, fogged = direct
    assert clear > 10
    assert fogged / clear == pytest.approx(math.exp(-1), abs=0.02)


def test_medium_denoiser_guide_is_scatter_albedo_with_zero_normal(tmp_path):
    from test_denoise import _render_guided, _SpyDenoiser

    spy = _SpyDenoiser()

    def build(scene):
        _fog_scene(4)(scene)
        scene.camera.set_fov(1)

    _render_guided(tmp_path, "medium_guides.png", spy, build)
    assert spy.calls
    _color, albedo, normal = spy.calls[0]
    h, w = albedo.shape[1:3]
    center = (slice(None), slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3))
    assert float(albedo[center].mean()) > 0.98
    assert float(normal[center].abs().max()) < 1e-6, "null shell is not a surface guide"


def test_medium_stack_overflow_is_reported_and_fails_closed(tmp_path):
    from algan import SETTINGS, Off, SceneManager

    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=2, max_bounces=0, denoise=False)
        with Scene(video_settings=SMOKE_TEST.set(resolution=(8, 8))) as scene:
            with Off():
                Scene.clear_lights()
                scene.set_background(WHITE)
                scene.camera.move_to(ORIGIN)
                scene.camera.look_at(-OUT)
                for size in range(2, 7):
                    shell = Prism(width=size, height=size, depth=size)
                    shell.set_material(
                        MeshPhysicalMaterial(
                            transmission=1, ior=1, roughness=0, sigma_s=0.1
                        )
                    )
                    shell.spawn(animate=False)
            result = scene.save_frame(tmp_path / "medium_overflow.png", overwrite=True)
        assert result.render_plan.truncations.medium_stack > 0
        assert result.render_plan.truncations.medium_query == 0
        from test_path_tracer import _read

        assert not _read(result)[..., :3].any(), (
            "unknown interior must not leak background"
        )
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)


@pytest.mark.parametrize(
    ("sigma_s", "sigma_a", "bounces"),
    [
        ((0.25, 0.25, 0.25), 0.0, 0),
        ((0.1, 0.3, 0.6), 0.2, 0),
        ((0.1, 0.3, 0.6), 0.2, 4),
        ((0.0, 0.0, 0.0), 0.25, 4),
    ],
)
def test_medium_coverage_matches_rgb_mean_ballistic_transmittance(
    tmp_path, sigma_s, sigma_a, bounces
):
    from test_path_tracer import _render_scene_exp

    from algan import RIGHT, TRANSPARENT

    def build(scene):
        Scene.clear_lights()
        scene.set_background(TRANSPARENT)
        scene.camera.set_fov(1)
        fog = Prism(width=8, height=8, depth=2)
        fog.set_material(
            MeshPhysicalMaterial(
                color=WHITE,
                transmission=1,
                ior=1,
                roughness=0,
                sigma_s=sigma_s,
                attenuation_color=(math.exp(-sigma_a),) * 3,
                attenuation_distance=1,
            )
        )
        fog.spawn(animate=False)
        # Keep medium tracking enabled for the purely absorbing test case,
        # without putting any scattering matter on its camera segment.
        remote = Prism(width=0.1, height=0.1, depth=0.1)
        remote.set_material(MeshPhysicalMaterial(transmission=1, ior=1, sigma_s=0.1))
        remote.move_to(RIGHT * 100)
        remote.spawn(animate=False)

    image = _render_scene_exp(
        tmp_path,
        f"medium_alpha_{bounces}_{sigma_a}_{sigma_s[0]}.png",
        build,
        256,
        video=SMOKE_TEST.set(resolution=(8, 8)),
        max_bounces=bounces,
        linear_color_space=False,
        tonemapping=False,
        experimental={
            "post_process_tonemap": False,
            "pt_error_target": 0,
            "pt_firefly_clamp": 0,
        },
    )
    expected = 1 - sum(math.exp(-2 * (sigma_a + ss)) for ss in sigma_s) / 3
    assert image.shape[-1] == 4
    assert not image[..., :3].any(), "unlit absorbing/scattering matter emits no light"
    assert float(image[..., 3].float().mean()) / 255 == pytest.approx(
        expected, abs=0.008
    )
