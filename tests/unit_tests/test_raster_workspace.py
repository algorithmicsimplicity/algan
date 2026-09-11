"""Real raster discovery can release raw records before resolving its sheets."""

from __future__ import annotations

import pytest
import torch

from algan import BLUE, LEFT, OUT, PREVIEW, RED, SETTINGS, UP, Cube, Off, Scene, Square
from algan.rendering.raytracing import raster_pipeline


@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("shadows", [False, True])
def test_raw_fragment_scratch_is_reclaimed_and_sheets_survive_reuse(
    monkeypatch, fresh_scene, frames, shadows
):
    SETTINGS.raytracing.set(shadows=shadows)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False)
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(32, 24), frames_per_second=2))
    with Off():
        cube = Cube().set_color(RED).set_opacity(0.6)
        cube.rotate(35, UP).spawn(animate=False)
        Square().set_color(BLUE).move(LEFT * 0.7 - OUT).spawn(animate=False)
    cube.rotate(10, UP)
    original = raster_pipeline.prepare_sparse_raster_coverage
    checked = []

    def compare_and_reclaim(*args, **kwargs):
        memory = args[4]
        before = memory.get_pointers()
        # The retained diagnostic form is the record-by-record oracle for
        # production ownership. Copy its answers before its arena scope exits.
        with memory.temp(clear_persist=True):
            retained = original(*args, **dict(kwargs, retain_fragments=True))
            if retained is None:
                return None
            expected = {
                name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for name, value in retained.items()
            }
            retained_bytes = before[1] - memory.current_reverse_pointer
        assert memory.get_pointers() == before
        actual = original(*args, **dict(kwargs, retain_fragments=False))
        assert actual is not None
        assert memory.current_pointer == before[0]
        assert not any(name.startswith("frag_") for name in actual)
        assert "run_offsets" not in actual
        raw_bytes = expected["num_fragments"] * 32 + (expected["num_covered"] + 1) * 4
        released = retained_bytes - (before[1] - memory.current_reverse_pointer)
        # Reverse alignment may shift by one word when raw records disappear.
        assert released >= raw_bytes - 8
        # Overwrite the now-free discovery workspace; every returned sheet
        # and covered-pixel buffer must live at the persistent end, not here.
        with memory.temp():
            size = min(1 << 20, memory.get_num_bytes_remaining())
            memory.get_tensor((size,), torch.uint8).fill_(255)
        for name, value in actual.items():
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(), expected[name]), name
            else:
                assert value == expected[name], name
        checked.append((expected["num_fragments"], released))
        return actual

    monkeypatch.setattr(
        raster_pipeline, "prepare_sparse_raster_coverage", compare_and_reclaim
    )
    images = []
    with torch.inference_mode():
        for batch in scene.get_frames(0, frames, post_processes=()):
            images.append(batch.detach().cpu().clone())
    assert checked, "the analytic raster route was not exercised"
    assert sum(image.shape[0] for image in images) == frames
    assert all(torch.isfinite(image).all() for image in images)


@pytest.mark.parametrize("stage", ["resolve", "drain"])
def test_sparse_memory_retry_shrinks_split_pool_and_preserves_frame(
    monkeypatch, fresh_scene, stage
):
    from algan import MeshStandardMaterial
    from algan.rendering.raytracing import tracer
    from algan.utils.memory_utils import InsufficientMemoryException

    SETTINGS.raytracing.set(shadows=True, max_bounces=2)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False)
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(32, 24), frames_per_second=1))
    with Off():
        cube = Cube().set_material(MeshStandardMaterial(metalness=1, roughness=0))
        cube.rotate(35, UP).spawn(animate=False)
        Square().set_color(BLUE).move(LEFT * 0.7 - OUT).spawn(animate=False)

    def render():
        with torch.inference_mode():
            return torch.cat(
                [
                    b.detach().cpu().clone()
                    for b in scene.get_frames(0, 1, post_processes=())
                ]
            )

    reference = render()
    capacities = []
    allocate = tracer._alloc_wavefront_state

    def track_pool(memory, slots, *args, **kwargs):
        capacities.append(slots)
        return allocate(memory, slots, *args, **kwargs)

    monkeypatch.setattr(tracer, "_alloc_wavefront_state", track_pool)
    target = raster_pipeline if stage == "resolve" else tracer
    name = (
        "shade_sparse_raster_coverage"
        if stage == "resolve"
        else "wavefront_traverse_events"
    )
    original = getattr(target, name)
    failures = []

    def fail_once(*args, **kwargs):
        if not failures:
            failures.append(1)
            raise InsufficientMemoryException("injected scratch exhaustion")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, fail_once)
    actual = render()
    assert failures, "the requested sparse stage was not reached"
    assert len(capacities) >= 2
    assert capacities[1] < capacities[0], "memory retry kept the exhausted split pool"
    assert (actual.to(torch.int16) - reference.to(torch.int16)).abs().max() <= 1
