"""Single-bit deferred shadow masks can use the existing hard-light kernel."""

from __future__ import annotations

import pytest
import torch

from algan import (
    OUT,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    WHITE,
    MeshLambertMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    Square,
)
from algan.rendering.raytracing import raster_taichi, shadow_queue
from algan.rendering.raytracing.tracer import _deferred_shadow_sample_count
from algan.scene_manager import SceneManager


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("columns", [3, 16])
@pytest.mark.parametrize("samples", [1, 4])
def test_only_compact_hard_lights_collapse(enabled, columns, samples):
    with SETTINGS.raytracing.experimental.override(
        shadow_deferred_single_sample=enabled
    ):
        assert _deferred_shadow_sample_count(samples, columns) == (
            1 if enabled and columns == 3 else samples
        )


@pytest.mark.parametrize("radius", [0.0, 0.2], ids=["hard", "soft"])
def test_zero_footprint_visibility_matches_every_original_sample(
    tmp_path, monkeypatch, radius
):
    original = raster_taichi.raster_shadow_trace
    original_make = shadow_queue.make_shadow_tracer
    params = raster_taichi._RASTER_SHADOW_TRACE_PARAMS
    checks = []
    saw_attenuation = []

    def checked(*inputs):
        if inputs[0] > 0 and not checks:
            args = list(inputs)
            args[params.index("event_dp")] = torch.zeros_like(
                args[params.index("event_dp")]
            )
            masks = args[params.index("event_msk")]
            # Reproduce the deferred event contract: only sample zero is
            # visible, and every sub-pixel offset is zero.
            args[params.index("event_msk")] = (masks & ~15) | 1
            for adaptive in (0, 1):
                args[params.index("sec_aa")] = 4
                args[params.index("adaptive_taps")] = adaptive
                original(*args)
                expected = args[params.index("shadow_vis")].clone()
                args[params.index("sec_aa")] = _deferred_shadow_sample_count(
                    4, args[params.index("light_col")].shape[2]
                )
                assert (args[params.index("sec_aa")] == 1) == (radius == 0)
                original(*args)
                actual = args[params.index("shadow_vis")]
                assert torch.equal(actual, expected)
                if radius > 0:
                    args[params.index("sec_aa")] = 1
                    original(*args)
                    assert not torch.equal(
                        args[params.index("shadow_vis")], expected
                    ), "soft-light fixture must demonstrate why the mask is retained"
                saw_attenuation.append(bool(((expected > 0) & (expected < 1)).any()))
                checks.append(adaptive)

    def checked_make(memory, sort_sources=None):
        launch = original_make(memory, sort_sources)

        def trace(*inputs):
            checked(*inputs)
            return launch(*inputs)

        return trace

    monkeypatch.setattr(shadow_queue, "make_shadow_tracer", checked_make)
    SceneManager.reset()
    try:
        with SETTINGS.override():
            SETTINGS.raytracing.shadows = True
            SETTINGS.raytracing.experimental.analytic_aa_secondary_samples = 4
            SETTINGS.raytracing.experimental.shadow_deferred_single_sample = True
            # This regression concerns the legacy mask-filtered fan. Budgeted
            # fans rotate through covered positions and need a different fixture.
            SETTINGS.raytracing.experimental.shadow_ray_budget = 0
            quality = SMOKE_TEST.set(resolution=(32, 32))
            with Scene(video_settings=quality) as scene:
                with Off():
                    scene.clear_lights()
                    PointLight(
                        location=RIGHT * 3 + OUT * 3,
                        color=WHITE,
                        shadow_radius=radius,
                        intensity=0.08,
                    ).spawn(animate=False)
                    Prism(width=6, height=6, depth=0.1).set_material(
                        MeshLambertMaterial(color=WHITE)
                    ).spawn(animate=False)
                    for opacity, distance in ((0.25, 1.0), (0.5, 1.1)):
                        Square(size=1.2, color=WHITE, opacity=opacity).move(
                            RIGHT * distance + OUT * distance
                        ).spawn(animate=False)
                scene.save_frame(
                    tmp_path / "sample_reuse", video_settings=quality, overwrite=True
                )
    finally:
        SceneManager.reset()
    assert checks == [0, 1]
    assert all(saw_attenuation), "fixture must exercise fractional shadow visibility"
