"""Late BVHs live in the batch region, never below discarded chunk records."""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE,
    LEFT,
    OUT,
    PREVIEW,
    SETTINGS,
    UP,
    Cube,
    MeshStandardMaterial,
    Off,
    Scene,
    Square,
)
from algan.rendering.raytracing import raster_pipeline, scene_builder, tracer
from algan.rendering.raytracing.truncation import record_path_samples, record_truncation
from algan.utils.memory_utils import InsufficientMemoryException, ManualMemory


def _render(scene, frames):
    with torch.inference_mode():
        return torch.cat(
            [
                part.detach().cpu().clone()
                for part in scene.get_frames(0, frames, post_processes=())
            ]
        )


@pytest.mark.parametrize("split", [False, True])
def test_late_build_restarts_clean_chunk_and_reuses_batch_trees(
    monkeypatch, fresh_scene, split
):
    from algan.rendering.memory_model import ChunkMemoryModel

    # Start with one two-frame renderer call; the split arm must exercise the
    # renderer's recursive retry, not the outer planner's one-frame default.
    monkeypatch.setattr(
        ChunkMemoryModel,
        "plan",
        lambda self, signature, requested, available: requested,
    )
    SETTINGS.raytracing.set(shadows=False, max_bounces=2)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False, bvh_defer=True)
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(24, 16), frames_per_second=2))
    with Off():
        cube = Cube().set_material(MeshStandardMaterial(metalness=1, roughness=0))
        cube.rotate(30, UP).spawn(animate=False)
        Square().set_color(BLUE).move(LEFT * 0.7 - OUT).spawn(animate=False)
    cube.rotate(10, UP)
    expected = _render(scene, 2)
    monkeypatch.setattr(scene_builder, "_bvh_deferral_eligible", lambda _: True)
    prepare = raster_pipeline.prepare_sparse_raster_coverage
    build = scene_builder.build_deferred_bvhs
    finish = tracer.post_process_frames
    prefill = tracer._prefill_background
    discoveries, builds, refills, completed = [], [], [], []
    attempted_split = []

    def prepare_and_record(
        merged, tri_screen, tri_bounds, bez_bounds, memory, *args, **kwargs
    ):
        before = memory.get_pointers()
        floor = merged[tracer.ARENA_RETAINED_REVERSE_POINTER]
        result = prepare(
            merged, tri_screen, tri_bounds, bez_bounds, memory, *args, **kwargs
        )
        assert memory.current_reverse_pointer < floor
        discoveries.append(
            (before, merged.get("bvh_deferred"), memory.current_reverse_pointer)
        )
        if merged.get("bvh_deferred"):
            # These belong to the discarded discovery, and must not enter the
            # accepted frame's plan. Perturb the background too, to prove the
            # restart refills it rather than accumulating twice into it.
            record_truncation("sheet_layers", 123, 16)
            record_path_samples(246, 2)
        return result

    def build_at_boundary(merged, memory=None):
        before = memory.get_pointers()
        assert discoveries
        assert discoveries[-1][1]
        assert before[1] == merged[tracer.ARENA_RETAINED_REVERSE_POINTER]
        assert before[1] > discoveries[-1][2], "coverage is still retained"
        assert before[0] < discoveries[-1][0][0], "discarded output was not reclaimed"
        result = build(merged, memory)
        builds.append((before, memory.get_pointers()))
        return result

    def record_prefill(*args, **kwargs):
        result = prefill(*args, **kwargs)
        if not refills:
            args[0].fill_(0.93)  # must not leak from the discarded attempt
        refills.append(1)
        return result

    def finish_or_split(memory, frames, **kwargs):
        if split and frames.shape[0] > 1 and not attempted_split:
            attempted_split.append(1)
            raise InsufficientMemoryException("injected chunk split after publication")
        completed.append(frames.shape[0])
        return finish(memory, frames, **kwargs)

    monkeypatch.setattr(
        raster_pipeline, "prepare_sparse_raster_coverage", prepare_and_record
    )
    monkeypatch.setattr(scene_builder, "build_deferred_bvhs", build_at_boundary)
    monkeypatch.setattr(tracer, "post_process_frames", finish_or_split)
    monkeypatch.setattr(tracer, "_prefill_background", record_prefill)
    actual = _render(scene, 2)
    assert len(builds) == 1, "trees should be published once, not once per chunk"
    assert sum(flag for _, flag, _ in discoveries) == 1
    assert len(refills) == len(discoveries)
    assert completed == ([1, 1] if split else [2])
    assert scene.last_render_plan.truncations.sheet_layers == 0
    assert scene.last_render_plan.path_samples_mean == 0
    assert (actual.to(torch.int16) - expected.to(torch.int16)).abs().max() <= 1


@pytest.mark.parametrize(
    "failure", [InsufficientMemoryException, RuntimeError, LookupError]
)
def test_boundary_publication_failure_is_transactional_and_bounded(
    monkeypatch, failure
):
    memory = ManualMemory(0, device=torch.device("cpu"), num_bytes=4096)
    persistent = memory.get_tensor((7,), torch.int64, persist=True)
    persistent.fill_(2**54 + 1)
    memory.get_tensor((3,), torch.int32).fill_(77)
    before = memory.get_pointers()
    merged = {"bvh_deferred": True, tracer.ARENA_RETAINED_REVERSE_POINTER: before[1]}
    attempts = []
    reclaimed = []

    def fail_publish(merged, memory):
        assert memory.get_pointers() == before
        attempts.append(merged["bvh_deferred"])
        merged["bvh_deferred"] = False
        merged["bvh_rehome_pending"] = True
        memory.get_tensor((13,), torch.int32).fill_(91)
        memory.get_tensor((29,), torch.int32, persist=True).fill_(93)
        raise failure("injected publication failure")

    monkeypatch.setattr(scene_builder, "build_deferred_bvhs", fail_publish)
    monkeypatch.setattr(
        tracer, "release_torch_memory", lambda **kwargs: reclaimed.append(1)
    )
    expected = (
        tracer.OutOfRenderMemory if failure is InsufficientMemoryException else failure
    )
    with pytest.raises(expected):
        tracer._publish_bvhs_at_chunk_boundary(merged, memory)
    assert len(attempts) == (2 if failure is InsufficientMemoryException else 1)
    assert len(reclaimed) == (1 if failure is InsufficientMemoryException else 0)
    assert memory.get_pointers() == before
    assert merged[tracer.ARENA_RETAINED_REVERSE_POINTER] == before[1]
    assert merged["bvh_rehome_pending"] is True
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(213)
    assert persistent.tolist() == [2**54 + 1] * 7
