"""End-to-end parity for both primary-visibility experiments, not benchmarks."""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    OUT,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    AmbientLight,
    Circle,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    Off,
    PointLight,
    Scene,
    TriangleMesh,
)
from algan.rendering.raytracing import raster_pipeline, tracer
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS


@pytest.mark.parametrize("shadows", [False, True])
def test_tiled_and_simple_paths_match_complete_renderer(monkeypatch, shadows):
    """Exercise independent toggles and repeated renders of preserved state.

    Includes full transparent layers, an opaque interior, a silhouette circuit
    and reflective/transmitting material. Compare pre-encoding frames, and spy
    on real launches so the tests cannot pass without using the new paths.
    """
    video = SMOKE_TEST.set(resolution=(48, 32), frames_per_second=1)
    SETTINGS.raytracing.set(
        samples_per_pixel=1,
        max_bounces=3,
        shadows=shadows,
        tonemapping=False,
        linear_color_space=False,
    )
    SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
    seen, drains, shadow_events = [], [], []
    prepare = raster_pipeline.prepare_sparse_raster_coverage
    drain = tracer.wavefront_shade
    trace_shadow = raster_pipeline.raster_shadow_trace

    def capture(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if result is not None:
            seen.append(
                {
                    k: result[k]
                    for k in (
                        "raster_tile_binning",
                        "raster_simple_interiors",
                        "num_simple_pixels",
                    )
                }
            )
        return result

    def capture_drain(*args, **kwargs):
        drains.append(True)
        return drain(*args, **kwargs)

    def capture_shadow(*args, **kwargs):
        shadow_events.append(True)
        return trace_shadow(*args, **kwargs)

    monkeypatch.setattr(raster_pipeline, "prepare_sparse_raster_coverage", capture)
    monkeypatch.setattr(tracer, "wavefront_shade", capture_drain)
    monkeypatch.setattr(raster_pipeline, "raster_shadow_trace", capture_shadow)
    SceneManager.reset()
    try:
        with Scene(video_settings=video) as scene:
            with Off():
                Scene.clear_lights()
                scene.set_background(BLACK)
                AmbientLight(color=WHITE, intensity=0.4).spawn(animate=False)
                PointLight(
                    location=RIGHT * 3 + UP * 3 + OUT * 5, color=WHITE, intensity=0.8
                ).spawn(animate=False)
                for z, color, alpha in (
                    (0.0, BLUE, 1.0),
                    (1.0, RED, 0.25),
                    (2.0, WHITE, 0.35),
                ):
                    panel = TriangleMesh(
                        vertices=[[-40.0, -40.0, z], [80.0, -40.0, z], [-40.0, 80.0, z]],
                        faces=[[0, 1, 2]],
                    )
                    panel.set_material(MeshLambertMaterial(color=color, opacity=alpha))
                    panel.spawn(animate=False)
                glass = TriangleMesh(
                    vertices=[[-0.8, -0.7, 2.5], [0.8, -0.7, 2.5], [0.0, 0.9, 2.5]],
                    faces=[[0, 1, 2]],
                )
                glass.set_material(
                    MeshPhysicalMaterial(
                        color=WHITE, roughness=0.12, transmission=0.4, ior=1.5
                    )
                )
                glass.spawn(animate=False)
                Circle(radius=0.25, color=RED).move(RIGHT * 1.3 + OUT * 3).spawn(
                    animate=False
                )
            outputs = []
            for bins, simple in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
                (True, True),
            ):
                with SETTINGS.raytracing.experimental.override(
                    raster_tile_binning=bins, raster_simple_interiors=simple
                ):
                    before = len(seen)
                    frame = torch.cat(
                        [batch.cpu() for batch in scene.get_frames(0, 1, post_processes=())]
                    )
                    assert len(seen) > before, (
                        "scene fell back from the analytic sparse frontend"
                    )
                    assert all(
                        row["raster_tile_binning"] == bins
                        and row["raster_simple_interiors"] == simple
                        for row in seen[before:]
                    )
                    if simple:
                        assert sum(row["num_simple_pixels"] for row in seen[before:]) > 0
                    outputs.append(frame.to(torch.int32))
            for result in outputs[1:]:
                assert result.shape == outputs[0].shape
                # Existing renderer tolerance: differing float reductions may
                # move the final conversion by a code value, not geometry.
                assert int((result - outputs[0]).abs().max()) <= 2
            assert drains, "reflective/transmitting scene never spawned continuations"
            assert bool(shadow_events) == shadows
    finally:
        SceneManager.reset()
