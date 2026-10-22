"""Two CUDA renders of a refractive, glossy scene are identical to the bit.

Glass in front of a rough reflective floor is where a CUDA render used to
differ from itself between runs. The refracted taps of one glass pixel that
meet the rough floor below all qualify for the pixel's one prefiltered glossy
event, and the first of them to reach an ``atomic_max`` took it: its
reflection direction, depth and blur radius became the whole pixel's glossy
reflection, and which one arrived first was GPU scheduling order (up to 240
levels on ``benchmarks/performance/graphics_scene.py``). The claim is now
settled by value after the shade launch (``gloss_claim_*`` in
``wavefront_kernels_taichi``). Beneath that, several rays of one pixel add into
its accumulator row with float ``atomic_add`` from one launch, which only
moved the last bit or two of the sum; a splitting batch now puts those
deposits on a fixed grid where the sum is exact (``_det_deposit``).

The frames are compared, and so is every tile's accumulator before it is
composited, entry by entry below the grid's exact range: an ulp of
reassociation there usually truncates away in the 8-bit frame, so a test of
the frames alone would pass by luck.

CUDA only: that is where the races were measured. Unmarked, so outside
``--fast``: only the renderer can break it, and the full suite runs after any
renderer change.
"""

from __future__ import annotations

from contextlib import closing

import pytest
import torch

from algan import (
    DOWN,
    GRAY_D,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    PREVIEW,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    DirectionalLight,
    Off,
    Prism,
    Scene,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.wavefront_kernels_taichi import _DET_SUM_SCALE
from algan.rendering.shaders.materials import (
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)
from algan.settings._startup import render_device

pytestmark = pytest.mark.skipif(
    render_device().type != "cuda", reason="run-to-run order only varies on CUDA"
)

VIDEO = PREVIEW.set(resolution=(320, 180))
FRAMES = 6
# Pins the frame windows, as the full-render harness does, so both renders
# batch alike and only the kernels' own order could tell them apart.
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024
# A sum of deposits on the grid is exact while it stays under 2^24 grid steps;
# a brighter one still rounds in arrival order (``_det_deposit``).
EXACT_BELOW = 2.0**24 / _DET_SUM_SCALE


def _build():
    """A glass ball and a mirror ball on a rough metal slab, the glass rising."""
    with Off():
        DirectionalLight(
            location=RIGHT * 4 + UP * 6 + OUT * 4, target=ORIGIN, intensity=2.0
        ).spawn(animate=False)
        ground = Prism(width=8, height=0.2, depth=8, color=GRAY_D).set_material(
            MeshStandardMaterial(metalness=0.55, roughness=0.18)
        )
        ground.move(DOWN * 1.1)
        glass = Sphere(radius=0.8, color=WHITE).set_material(
            MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)
        )
        glass.move(LEFT * 0.7 + DOWN * 0.2 + OUT * 0.5)
        chrome = Sphere(radius=0.6, color=WHITE).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.05)
        )
        chrome.move(RIGHT * 0.9 + DOWN * 0.3 + IN * 0.6)
        for mob in (ground, glass, chrome):
            mob.spawn(animate=False)
        camera = Scene.get_camera()
        camera.move(UP * 1.2 + OUT * 0.5).look_at(DOWN * 0.4)
    with Sync(runtime=FRAMES / VIDEO.frames_per_second):
        glass.move(UP * 0.3)


def _render(scene, accumulators):
    accumulators.clear()
    with closing(scene.get_frames(0, FRAMES, post_processes=())) as stream:
        frames = torch.cat([batch.cpu() for batch in stream])
    return frames, list(accumulators)


def test_a_refractive_glossy_scene_renders_identically_twice(monkeypatch):
    accumulators = []
    gloss_scatter = tracer.gloss_scatter

    def recording_scatter(*args, **kwargs):
        # ``pix_accum`` (plain and glossy rows) of one frame-part of a tile,
        # just after its drain and before anything reads it.
        accumulators.append(args[7].detach().to("cpu", copy=True))
        return gloss_scatter(*args, **kwargs)

    claims = [0]
    decide = tracer.gloss_claim_decide

    def counting_decide(*args, **kwargs):
        claims[0] += 1
        return decide(*args, **kwargs)

    monkeypatch.setattr(tracer, "gloss_scatter", recording_scatter)
    monkeypatch.setattr(tracer, "gloss_claim_decide", counting_decide)
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.computing.set(
            available_memory_override=AVAILABLE_MEMORY_OVERRIDE, torch_compile=False
        )
        with Scene(video_settings=VIDEO) as scene:
            _build()
            first, first_accum = _render(scene, accumulators)
            second, second_accum = _render(scene, accumulators)
    finally:
        SETTINGS.restore(snapshot)

    # Otherwise the scene no longer reaches what this test is about.
    assert first_accum, "the split-sum glossy route did not run"
    assert claims[0] > 0, "no drain iteration settled a glossy claim"

    assert len(first_accum) == len(second_accum)
    entries = 0
    for a, b in zip(first_accum, second_accum):
        exact = (a.abs() < EXACT_BELOW) & (b.abs() < EXACT_BELOW)
        entries += int(((a != b) & exact).sum())
    diff = (first.int() - second.int()).abs()
    report = (
        f"two renders differ: {int((diff > 0).sum())} frame channels, by up "
        f"to {int(diff.max())} levels; {entries} accumulator entries"
    )
    assert torch.equal(first, second), report
    assert entries == 0, report
