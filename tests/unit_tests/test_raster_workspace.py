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


@pytest.mark.parametrize("stage", ["resolve", "drain", "readback", "composite"])
def test_sparse_attempt_exception_restores_scope_and_retains_late_bvh(
    monkeypatch, fresh_scene, stage
):
    from contextlib import contextmanager

    from algan import MeshStandardMaterial
    from algan.rendering.raytracing import scene_builder, tracer
    from algan.utils.memory_utils import ManualMemory

    # Deliberately simulate a false-positive deferral eligibility decision.
    # The runtime contract promises to build real trees on the first spawned
    # continuation, after this attempt has already reserved per-pixel state.
    monkeypatch.setattr(scene_builder, "_bvh_deferral_eligible", lambda _: True)
    SETTINGS.raytracing.set(shadows=False, max_bounces=2)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False, bvh_defer=True)
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(24, 16), frames_per_second=1))
    with Off():
        Cube().set_material(MeshStandardMaterial(metalness=1, roughness=0)).rotate(
            30, UP
        ).spawn(animate=False)
        Square().set_color(BLUE).move(LEFT * 0.7 - OUT).spawn(animate=False)

    scopes, built, failures = [], [], []
    original_temp = ManualMemory.temp
    original_build = scene_builder.build_deferred_bvhs

    @contextmanager
    def track_temp(memory, *args, **kwargs):
        # Only sparse attempt scopes use a callable retention floor. Observe
        # cleanup before the enclosing render/job handlers can reset the arena.
        if callable(kwargs.get("persist_floor")):
            before = memory.get_pointers()
            try:
                with original_temp(memory, *args, **kwargs):
                    yield
            finally:
                floor = kwargs["persist_floor"]()
                expected_reverse = (
                    min(before[1], floor) if floor is not None else before[1]
                )
                scopes.append((before, memory.get_pointers()))
                assert memory.get_pointers() == (before[0], expected_reverse)
                with original_temp(memory, clear_persist=True):
                    size = min(1 << 20, memory.get_num_bytes_remaining() // 2)
                    memory.get_tensor((size,), torch.uint8).fill_(193)
                    memory.get_tensor((size,), torch.uint8, persist=True).fill_(195)
                for tensor, expected in built:
                    assert torch.equal(
                        tensor.cpu().contiguous().view(torch.uint8), expected
                    )
        else:
            with original_temp(memory, *args, **kwargs):
                yield

    def capture_build(merged, memory=None):
        result = original_build(merged, memory)
        for name in ("tri_bvh", "bez_bvh", "tri_opaque_bvh", "bez_opaque_bvh"):
            tree = merged[name]
            for field in ("blocks", "node_miss", "leaf_prim", "leaf_tspan"):
                tensor = getattr(tree, field)
                if tensor.numel():
                    built.append(
                        (tensor, tensor.cpu().contiguous().view(torch.uint8).clone())
                    )
        return result

    name = {
        "resolve": "shade_sparse_raster_coverage",
        "drain": "wavefront_traverse_events",
        "readback": "_read_tile_alloc",
        "composite": "wf_composite_accum_sparse",
    }[stage]
    target = raster_pipeline if stage == "resolve" else tracer

    original = getattr(target, name)

    def fail(*args, **kwargs):
        if stage == "readback" and not built:
            return original(*args, **kwargs)
        failures.append(stage)
        raise LookupError(f"injected sparse {stage}")

    monkeypatch.setattr(ManualMemory, "temp", track_temp)
    monkeypatch.setattr(scene_builder, "build_deferred_bvhs", capture_build)
    monkeypatch.setattr(target, name, fail)

    def render():
        with torch.inference_mode():
            return list(scene.get_frames(0, 1, post_processes=()))

    with pytest.raises(LookupError, match=f"injected sparse {stage}"):
        render()
    assert failures == [stage]
    assert scopes
    if stage != "resolve":
        assert built, "a real late BVH was not built inside the attempt"


@pytest.mark.parametrize("failure_stage", ["rehome", "drain"])
def test_late_bvh_memory_retry_rebinds_and_matches_clean_render(
    monkeypatch, fresh_scene, failure_stage
):
    from algan import MeshStandardMaterial
    from algan.rendering.raytracing import scene_builder, tracer
    from algan.utils.memory_utils import InsufficientMemoryException

    SETTINGS.raytracing.set(shadows=False, max_bounces=2)
    SETTINGS.raytracing.experimental.set(glossy_reflection=False, bvh_defer=True)
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(24, 16), frames_per_second=1))
    with Off():
        Cube().set_material(MeshStandardMaterial(metalness=1, roughness=0)).rotate(
            30, UP
        ).spawn(animate=False)
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
    # Compare eager construction against a deliberately forced late build.
    monkeypatch.setattr(scene_builder, "_bvh_deferral_eligible", lambda _: True)
    failures, builds = [], []
    build = scene_builder.build_deferred_bvhs
    copy = scene_builder._copy_merged_scene_to_arena
    traverse = tracer.wavefront_traverse_events

    def build_and_track(merged, memory=None):
        builds.append(
            (merged.get("bvh_deferred"), merged.get("bvh_rehome_pending", False))
        )
        result = build(merged, memory)
        assert not merged.get("bvh_rehome_pending")
        return result

    def copy_and_fail(group, memory, *args, **kwargs):
        if (
            failure_stage == "rehome"
            and set(group) <= set(scene_builder._DEFERRED_BVH_KEYS)
            and not failures
        ):
            # Fail after a partial reverse allocation, not before the allocator
            # is touched. The next attempt must retry publication, not traverse
            # a stale placeholder or an ordinary allocator-backed tree.
            memory.get_tensor((31,), torch.int32, persist=True).fill_(123)
            failures.append(failure_stage)
            raise InsufficientMemoryException("injected BVH publication failure")
        return copy(group, memory, *args, **kwargs)

    def traverse_and_fail(*args, **kwargs):
        if failure_stage == "drain" and not failures:
            failures.append(failure_stage)
            raise InsufficientMemoryException("injected post-build drain failure")
        return traverse(*args, **kwargs)

    monkeypatch.setattr(scene_builder, "build_deferred_bvhs", build_and_track)
    monkeypatch.setattr(scene_builder, "_copy_merged_scene_to_arena", copy_and_fail)
    monkeypatch.setattr(tracer, "wavefront_traverse_events", traverse_and_fail)
    actual = render()
    assert failures == [failure_stage]
    assert builds[0][0], "the first tree build was not deferred"
    if failure_stage == "rehome":
        assert (False, True) in builds, "publication was not retried after construction"
    assert (actual.to(torch.int16) - reference.to(torch.int16)).abs().max() <= 1


@pytest.mark.parametrize("opaque", [False, True])
def test_discovery_reuses_runs_and_rebuilds_only_after_truncation(
    monkeypatch, fresh_scene, opaque
):
    """Exercise the production ownership transition, not just helper arguments."""
    from algan.rendering.raytracing.pixel_runs import PixelRunCSR

    SETTINGS.raytracing.set(shadows=False)
    SETTINGS.raytracing.experimental.set(
        analytic_aa_one_mesh=True, glossy_reflection=False
    )
    scene = Scene.current()
    scene.set_video_settings(PREVIEW.set(resolution=(24, 16), frames_per_second=1))
    with Off():
        for depth in (0, -0.5):
            Cube().set_color(RED).set_opacity(1.0 if opaque else 0.5).move(
                OUT * depth
            ).spawn(animate=False)
    build = PixelRunCSR.from_sorted_pixels
    prefix = raster_pipeline._opaque_prefix_keep
    one_mesh = raster_pipeline._one_mesh_pixel_caps
    builds, cuts, consumers = [], [], []

    def track_build(cls, pixels, **kwargs):
        result = build(pixels, **kwargs)
        builds.append(result)
        return result

    def track_prefix(flags, counts, n, *, runs=None):
        assert runs is builds[-1]
        keep = prefix(flags, counts, n, runs=runs)
        cuts.append(int(keep.sum()) != n)
        return keep

    def track_one_mesh(*args, runs=None):
        assert runs is builds[-1]
        assert runs.num_fragments == args[0].numel()
        consumers.append(runs)
        return one_mesh(*args, runs=runs)

    monkeypatch.setattr(PixelRunCSR, "from_sorted_pixels", classmethod(track_build))
    monkeypatch.setattr(raster_pipeline, "_opaque_prefix_keep", track_prefix)
    monkeypatch.setattr(raster_pipeline, "_one_mesh_pixel_caps", track_one_mesh)
    with torch.inference_mode():
        frames = list(scene.get_frames(0, 1, post_processes=()))
    assert frames
    assert consumers
    assert len(builds) == len(consumers) + sum(cuts)
    if opaque:
        assert any(cuts), "fixture must exercise membership-changing truncation"
    else:
        assert not cuts
