"""Prepared batches share immutable allocation and execution decisions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.wavefront_kernels_taichi import sca_width


@pytest.fixture
def scene_facts():
    SETTINGS.raytracing.set(samples_per_pixel=1, shadows=False, analytic_aa=True)
    SETTINGS.raytracing.experimental.set(
        hybrid_raster=True,
        sheet_resolve=True,
        analytic_aa_run=True,
        raster_sparse_coverage=True,
        raster_empty_skip=True,
        raster_covered_shade=True,
        wf_mem_trim=False,
    )
    return {
        "num_triangles": 1,
        "num_circuits": 0,
        "tri_frame_valid": torch.ones((1, 1), dtype=torch.bool),
        "has_user_pipeline": False,
        "has_custom_scatter": False,
        "has_refractive": False,
        "has_refl_transparent": False,
        "has_any_translucent": False,
        "all_visible_opaque": True,
        "has_any_opaque": True,
        "tri_material_ids": (0,),
    }


def test_policy_is_frozen_and_new_render_uses_live_settings(scene_facts):
    first = tracer.resolve_batch_policy(scene_facts, 4)
    assert first.primary_route == "analytic_sheets"
    assert first.effective_aa == first.frame_scale == first.kernel_aa == 1
    assert first.fallback_reasons == ()
    with pytest.raises(FrozenInstanceError):
        first.effective_aa = 9
    SETTINGS.raytracing.set(analytic_aa=False)
    SETTINGS.raytracing.experimental.set(inplace_aa=False)
    second = tracer.resolve_batch_policy(scene_facts, 4)
    assert second is not first
    assert second.primary_route == "classic_wavefront"
    assert second.frame_scale == second.effective_aa == 4
    assert second.kernel_aa == 1
    assert "analytic_aa_disabled" in second.fallback_reasons
    assert first.primary_route == "analytic_sheets"
    SETTINGS.raytracing.experimental.set(inplace_aa=True)
    third = tracer.resolve_batch_policy(scene_facts, 4)
    assert third.frame_scale == 1
    assert third.kernel_aa == third.effective_aa == 4
    assert not third.wavefront.fused_generation


@pytest.mark.parametrize(
    "gate",
    [
        "hybrid_raster",
        "sheet_resolve",
        "analytic_aa_run",
        "raster_sparse_coverage",
        "raster_empty_skip",
        "raster_covered_shade",
    ],
)
def test_policy_names_each_sheet_route_veto(scene_facts, gate):
    SETTINGS.raytracing.experimental.set(**{gate: False})
    policy = tracer.resolve_batch_policy(scene_facts, 3)
    assert policy.primary_route == "classic_wavefront"
    assert f"{gate}_disabled" in policy.fallback_reasons
    assert policy.effective_aa == 3


@pytest.mark.parametrize(
    ("change", "kwargs", "reason"),
    [
        ({"tri_frame_valid": None}, {}, "analytic_projection_unavailable"),
        ({}, {"near_clip": 0.1}, "near_clipping"),
        ({"num_triangles": 0}, {}, "no_raster_geometry"),
        ({"has_custom_scatter": True}, {}, "custom_scatter"),
        (
            {},
            {"environment_map": object(), "transparent_background": True},
            "transparent_environment_background",
        ),
    ],
)
def test_policy_reports_data_and_capability_fallbacks(
    scene_facts, change, kwargs, reason
):
    scene_facts.update(change)
    SETTINGS.raytracing.experimental.set(fragment_shading=True)
    policy = tracer.resolve_batch_policy(scene_facts, 2, **kwargs)
    assert reason in policy.fallback_reasons
    assert policy.primary_route == "classic_wavefront"
    plan = tracer._build_render_plan(1, None, scene_facts, execution_policy=policy)
    assert plan.as_dict()["primary_route"] == policy.primary_route
    assert plan.as_dict()["effective_anti_alias_level"] == 2
    assert reason in plan.as_dict()["fallback_reasons"]


def test_path_policy_retains_shadow_request_without_deterministic_state(scene_facts):
    SETTINGS.raytracing.set(samples_per_pixel=3, shadows=True, max_bounces=7)
    policy = tracer.resolve_batch_policy(scene_facts, 8)
    assert policy.primary_route == "path_tracer"
    assert policy.wavefront is None
    assert policy.samples_per_pixel == 3
    assert policy.max_bounces == 7
    assert policy.effective_aa == policy.frame_scale == policy.kernel_aa == 1
    assert policy.shadows
    assert policy.shadow_mode == 0
    assert policy.fallback_reasons == ()


def test_policy_preserves_continuation_width_and_extended_light_forcing(scene_facts):
    SETTINGS.raytracing.experimental.set(fragment_shading=False)
    scene_facts["has_refractive"] = True
    policy = tracer.resolve_batch_policy(
        scene_facts, light_sources=[SimpleNamespace(_render_aux=object())]
    )
    assert policy.fragment_shading
    assert policy.lights_extended
    assert policy.wavefront.refraction
    assert policy.wavefront.state_scalar_width == sca_width(policy.wavefront.ior_stack)
    assert not policy.wavefront.opaque_closest
    assert policy.wavefront.pool_ratio > 1


def test_preflight_and_execution_reuse_one_policy_then_refresh_next_render(
    monkeypatch, fresh_scene
):
    from algan import BLUE, PREVIEW, Off, Scene, Square
    from algan.rendering.raytracing import settings as rt

    SETTINGS.raytracing.set(shadows=False, max_bounces=0)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False)
    scene = Scene.current()
    scene.set_video_settings(
        PREVIEW.set(resolution=(24, 16), frames_per_second=1, supersampling=2)
    )
    with Off():
        Square().set_color(BLUE).spawn(animate=False)
    prepared, executed, validated = [], [], []
    resolve = tracer.resolve_batch_policy
    wavefront = tracer.raytrace_render_wavefront
    validate = tracer._validate_render_capabilities

    def prepare(*args, **kwargs):
        policy = resolve(*args, **kwargs)
        prepared.append(policy)
        return policy

    def execute(*args, **kwargs):
        executed.append(kwargs["policy"])
        return wavefront(*args, **kwargs)

    def validation(*args, **kwargs):
        validated.append(kwargs["execution_policy"])
        return validate(*args, **kwargs)

    monkeypatch.setattr(tracer, "resolve_batch_policy", prepare)
    monkeypatch.setattr(tracer, "raytrace_render_wavefront", execute)
    monkeypatch.setattr(tracer, "_validate_render_capabilities", validation)
    for analytic in (True, False):
        SETTINGS.raytracing.set(analytic_aa=analytic)
        with torch.inference_mode():
            frames = [
                b.detach().cpu().clone()
                for b in scene.get_frames(0, 1, post_processes=())
            ]
        assert frames
        assert frames[0].shape[1:3] == (16, 24)
        assert prepared[-1].analytic_raster is analytic
        assert validated[-1] is prepared[-1]
        assert executed[-1] is prepared[-1].wavefront
    assert len(prepared) == len(executed) == len(validated) == 2
    assert prepared[0] is not prepared[1]
    assert rt.analytic_aa is False


@pytest.mark.parametrize(
    ("enabled", "anyhit", "facts", "expected"),
    [
        (False, True, {}, 0),
        (True, False, {}, 1),
        (True, "gather", {}, 4),
        (True, True, {}, 1),  # unknown transmission remains conservative
        (True, True, {"has_transmissive": True}, 1),
        (True, True, {"has_transmissive": False}, 2),
        (
            True,
            True,
            {
                "has_transmissive": False,
                "tri_has_translucent": False,
                "bez_has_translucent": False,
                "has_uncertain_texture_alpha": True,
            },
            2,
        ),
        (
            True,
            True,
            {
                "has_transmissive": False,
                "tri_has_translucent": False,
                "bez_has_translucent": False,
                "has_uncertain_texture_alpha": False,
            },
            3,
        ),
    ],
)
def test_policy_preserves_all_shadow_capability_modes(
    scene_facts, enabled, anyhit, facts, expected
):
    scene_facts.update(facts)
    SETTINGS.raytracing.set(shadows=enabled)
    SETTINGS.raytracing.experimental.set(shadow_anyhit=anyhit)
    policy = tracer.resolve_batch_policy(scene_facts)
    assert policy.shadows is enabled
    assert policy.shadow_mode == expected


@pytest.mark.parametrize("nested", [False, True])
def test_policy_ior_width_is_fixed_for_its_prepared_batch(
    scene_facts, monkeypatch, nested
):
    from algan.rendering.raytracing import settings as rt

    scene_facts["has_refractive"] = True
    monkeypatch.setattr(rt, "nested_ior", nested)
    policy = tracer.resolve_batch_policy(scene_facts)
    assert policy.wavefront.refraction
    assert policy.wavefront.ior_stack is nested
    assert policy.wavefront.state_scalar_width == sca_width(nested)
    monkeypatch.setattr(rt, "nested_ior", not nested)
    assert policy.wavefront.state_scalar_width == sca_width(nested)
    assert tracer.resolve_batch_policy(scene_facts).wavefront.ior_stack is not nested
