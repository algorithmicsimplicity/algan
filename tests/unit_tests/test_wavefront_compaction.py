import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer
from algan.utils.memory_utils import ManualMemory


def _fake_compact(
    source,
    num_source,
    scan_pool,
    desired_status,
    rs_int,
    rs_key,
    use_key,
    desired_key,
    output,
    output_count,
):
    indexes = (
        torch.arange(num_source, dtype=torch.int32)
        if scan_pool
        else source[:num_source]
    )
    keep = rs_int[indexes.long(), 2] == desired_status
    if use_key:
        keep &= rs_key[indexes.long()] == desired_key
    selected = indexes[keep]
    output[: selected.numel()].copy_(selected)
    output_count[0] = selected.numel()


def test_arena_compactor_filters_previous_set_and_scans_split_pool(monkeypatch):
    monkeypatch.setattr(tracer, "compact_ray_slots", _fake_compact)
    memory = ManualMemory(0, device="cpu", num_bytes=256)
    before = memory.current_pointer
    compactor = tracer._ArenaRayCompactor(memory, 4)

    # Two capacity-sized index arrays plus the one-word count.  A key-filter
    # placeholder is unnecessary because that argument compiles out.
    assert memory.current_pointer - before == 2 * 4 * 4 + 4

    status = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    active = compactor.initial(3)
    got = compactor.select(status, 0, source=active)
    assert torch.equal(got, torch.tensor([0, 2], dtype=torch.int32))

    # A splitting path must rediscover the newly activated spare slot 3.
    status[0, 2] = 1
    got = compactor.select(status, 0, source=got, scan_pool=True)
    assert torch.equal(got, torch.tensor([2, 3], dtype=torch.int32))

    keys = torch.tensor([5, 5, 7, 7], dtype=torch.int32)
    got = compactor.select(
        status, 0, scan_pool=True, rs_key=keys, desired_key=7
    )
    assert torch.equal(got, torch.tensor([2, 3], dtype=torch.int32))


def test_auto_tile_size_accounts_for_alignment_and_fixed_words(monkeypatch):
    split_k = 4
    wanted = 7
    fixed = 4 * torch.int32.itemsize
    per_primary = tracer._wavefront_state_bytes_per_primary(split_k)
    # Start the tile after one uint8 byte. ManualMemory must spend three bytes
    # aligning the first f32 state tensor.
    total = 1 + 3 + fixed + wanted * per_primary
    memory = ManualMemory(0, device="cpu", num_bytes=total)
    memory.get_tensor((1,), torch.uint8)

    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_AUTO", True)
    # Values above 1 are clamped: an environment override must never grant
    # permission to size beyond the actual arena.
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_SAFETY", 1.5)
    # Deliberately put the exact fit below the preferred floor: the floor must
    # not turn into an arena overcommit.
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_MIN", 1 << 18)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_MAX", 1 << 25)

    got = tracer._auto_primary_per_tile(
        memory, split_k, static_primary=100,
        fixed_bytes=fixed,
    )
    assert got == wanted

    pool = got * split_k
    tracer._alloc_wavefront_state(memory, pool, 7)
    memory.get_tensor((pool,), torch.int32)  # rs_pix
    memory.get_tensor((got, 7), torch.float32)  # pix_accum
    memory.get_tensor((2,), torch.int32)  # rs_alloc
    memory.get_tensor((1,), torch.int32)  # rs_vis
    tracer._ArenaRayCompactor(memory, pool)
    assert memory.get_num_bytes_remaining() == 0


def test_fixed_wavefront_estimator_uses_configured_route_targets(monkeypatch):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 1)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_AUTO", False)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_RAYS", 12)
    monkeypatch.setattr(rt_settings, "FRAGMENT_SHADING", True)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    monkeypatch.setattr(rt_settings, "WF_MEM_TRIM", False)
    monkeypatch.setattr(rt_settings, "WF_GEN_FUSED", False)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", False)
    monkeypatch.setattr(tracer, "_scene_has_custom_scatter", lambda _scene: False)

    scene = {
        "has_user_pipeline": False,
        "has_refractive": False,
        "textured_active": False,
    }
    tail = 7 * torch.float32.itemsize
    base_slot = tracer._wavefront_state_bytes_per_primary(1) - tail

    general = tracer.get_wavefront_memory_required(scene, 1, 20, 1)
    # col_row + gen_meta + two layer offsets + rs_vis + compactor count +
    # two shared allocator words.
    assert general == 12 * (base_slot + tail) + 32

    scene["textured_active"] = True
    textured = tracer.get_wavefront_memory_required(scene, 1, 20, 1)
    # Textured route: gen_meta, compactor count, and shared allocator.
    assert textured == 12 * (base_slot + tail) + 16

    scene["textured_active"] = False
    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", True)
    monkeypatch.setattr(rt_settings, "SHADOWS", True)
    sorted_bytes = tracer.get_wavefront_memory_required(scene, 1, 20, 1)
    sorted_primary = 8  # (12 * 2) // 3
    event_slot = 16 * 4 + 3 * 4
    # Sorted route: per-slot event/shadow state plus four fixed words.
    assert sorted_bytes == (
        sorted_primary * (base_slot + event_slot + tail) + 16
    )

    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", False)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    scene["has_refractive"] = True
    split = int(rt_settings.REFRACT_SPLIT_SLOTS)
    split_primary = max(1, 12 // split)
    split_bytes = tracer.get_wavefront_memory_required(scene, 1, 20, 1)
    assert split_bytes == (
        split_primary * split * base_slot + split_primary * tail + 32
    )


def test_auto_wavefront_estimator_reserves_one_primary_for_each_route(
    monkeypatch,
):
    monkeypatch.setattr(rt_settings, "SAMPLES_PER_PIXEL", 1)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_AUTO", True)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_TILE_RAYS", 12)
    monkeypatch.setattr(rt_settings, "FRAGMENT_SHADING", True)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    monkeypatch.setattr(rt_settings, "WF_MEM_TRIM", False)
    monkeypatch.setattr(rt_settings, "WF_GEN_FUSED", False)
    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", False)
    monkeypatch.setattr(tracer, "_scene_has_custom_scatter", lambda _scene: False)

    scene = {
        "has_user_pipeline": False,
        "has_refractive": False,
        "textured_active": False,
    }
    tail = 7 * torch.float32.itemsize
    base_slot = tracer._wavefront_state_bytes_per_primary(1) - tail

    # General route: one pool slot, one primary tail, and all general metadata.
    assert tracer.get_wavefront_memory_required(scene, 1, 20, 1) == (
        base_slot + tail + 32
    )

    # Textured route has the same one-slot state and only gen/count metadata.
    scene["textured_active"] = True
    assert tracer.get_wavefront_memory_required(scene, 1, 20, 1) == (
        base_slot + tail + 16
    )

    # Sorted shadows add the full event record/key/primitive/visibility state
    # for that one pool slot.
    scene["textured_active"] = False
    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", True)
    monkeypatch.setattr(rt_settings, "SHADOWS", True)
    event_slot = 16 * 4 + 3 * 4
    assert tracer.get_wavefront_memory_required(scene, 1, 20, 1) == (
        base_slot + event_slot + tail + 16
    )

    # A single refractive primary still requires its complete split-slot pool.
    monkeypatch.setattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", False)
    monkeypatch.setattr(rt_settings, "SHADOWS", False)
    scene["has_refractive"] = True
    split = int(rt_settings.REFRACT_SPLIT_SLOTS)
    assert tracer.get_wavefront_memory_required(scene, 1, 20, 1) == (
        split * base_slot + tail + 32
    )
