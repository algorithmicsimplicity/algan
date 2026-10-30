"""The sheet route sizes its tiles from what the render job's tiles have used
(``ALGAN_OPT_DISABLE=adaptivepool`` sizes them from the estimate, as before).

``tracer._TileDemand`` keeps, per render job and per pool estimate, the most
slots per primary and the most scratch bytes per slot any planned-size tile
attempt has shown; each later chunk takes its pool ratio and a scratch reserve
from it. Held here: the record's arithmetic, that measuring a tile leaves the
chunk's arena high-water mark as the chunk model would have read it, and --
one child process per arm, on one CPU thread, where a render through glass and
over a mirror is reproducible -- that a render planned from the record is
byte-identical to one sized from the estimate, while it plans from a smaller
ratio and needs fewer out-of-memory retries. (With the drain's ray sort pinned
off: see ``_render_arm``. Any change of tile size moves a splitting pixel's
last bits through it, and on a GPU those bits race anyway.)

Outside the fast suite: only a change to the sheet route's tile driver, or to
what a chunk holds while its tiles run (``_BUDGET``), can break it.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from algan.animation_timeline import timeline as timeline_module
from algan.rendering.raytracing import settings as rt_module
from algan.rendering.raytracing import tracer
from algan.utils.memory_utils import ManualMemory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORT = "adaptive-tile-pool report: "
_FRAMES = 4


def test_the_record_starts_from_the_estimate():
    demand = tracer._TileDemand()
    assert demand.pool_ratio(8) == 8
    assert demand.pool_ratio(1) == 1
    assert demand.scratch_reserve() == 0


def test_the_ratio_follows_the_largest_demand_with_headroom():
    demand = tracer._TileDemand()
    demand.observe_slots(1000, 2200)
    assert demand.pool_ratio(8) == math.ceil(2.2 * tracer._POOL_DEMAND_MARGIN)
    # A smaller reading changes nothing.
    demand.observe_slots(1000, 1500)
    assert demand.slots_per_primary == 2.2
    assert demand.pool_ratio(8) == 3
    # A batch that does not split keeps its one slot a primary.
    assert demand.pool_ratio(1) == 1


def test_the_ratio_never_drops_below_two_or_climbs_past_the_estimate():
    lone = tracer._TileDemand()
    lone.observe_slots(1000, 1000)
    assert lone.pool_ratio(8) == 2
    heavy = tracer._TileDemand()
    heavy.observe_slots(10, 1000)
    assert heavy.pool_ratio(8) == 8
    assert heavy.pool_ratio(2) == 2


@pytest.mark.parametrize(
    ("attempt", "planned", "speaks"),
    [(1000, 1000, True), (984, 1969, True), (983, 1969, False), (1, 4000, False)],
)
def test_only_a_planned_size_attempt_is_recorded(attempt, planned, speaks):
    assert tracer._plans_like(attempt, planned) is speaks


def test_an_attempt_out_of_arena_records_a_lower_bound(monkeypatch):
    monkeypatch.setattr(tracer, "_SCRATCH_SAMPLE_PRIMARIES", 8)
    memory = ManualMemory(0, device="cpu", num_bytes=4096)
    memory.get_tensor((256,), torch.float32)  # the tile's state: 1024 bytes
    peak = tracer._TileScratchPeak(memory)
    demand = tracer._TileDemand()
    tracer._note_scratch_overrun(demand, peak, memory, 16, 16, 40)
    # More than every byte left beside the state, over at most the pool.
    assert demand.scratch_per_slot == (len(memory) - 1024) / 40
    # Not from an attempt cut below half the plan, nor where no peak began.
    smaller = tracer._TileDemand()
    tracer._note_scratch_overrun(smaller, peak, memory, 16, 64, 40)
    tracer._note_scratch_overrun(smaller, None, memory, 16, 16, 40)
    assert smaller.scratch_per_slot == 0.0


def test_the_scratch_reserve_follows_the_largest_rate_per_slot():
    demand = tracer._TileDemand()
    demand.observe_scratch(900_000, 3000)
    demand.observe_scratch(1_000_000, 4000)
    assert demand.scratch_per_slot == 300
    assert demand.scratch_reserve() == math.ceil(300 * tracer._SCRATCH_DEMAND_MARGIN)


def test_the_record_is_the_jobs_and_kept_per_estimate(monkeypatch):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset())
    memory = ManualMemory(0, device="cpu", num_bytes=4096)
    glass = tracer._tile_demand(memory, 8)
    assert glass is tracer._tile_demand(memory, 8)
    assert tracer._tile_demand(memory, 1) is not glass
    # A new job's arena starts over.
    assert (
        tracer._tile_demand(ManualMemory(0, device="cpu", num_bytes=4096), 8)
        is not glass
    )
    unmanaged = ManualMemory(0, device="cpu", num_bytes=4096, managed=False)
    assert tracer._tile_demand(unmanaged, 8) is None
    # Fixed tiles were asked for: nothing to plan.
    monkeypatch.setattr(rt_module, "wavefront_tile_auto", False)
    assert tracer._tile_demand(memory, 8) is None
    monkeypatch.setattr(rt_module, "wavefront_tile_auto", True)
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({"adaptivepool"}))
    assert tracer._tile_demand(memory, 8) is None


@pytest.mark.parametrize("earlier_peak", [0, 3000], ids=["new-high", "old-high"])
def test_measuring_a_tile_leaves_the_chunks_high_water_mark(earlier_peak):
    memory = ManualMemory(0, device="cpu", num_bytes=4096)
    memory.get_tensor((64,), torch.float32)  # the tile's state: 256 bytes
    memory.max_pointer = max(memory.max_pointer, earlier_peak)
    before = memory.max_pointer
    peak = tracer._TileScratchPeak(memory)
    with memory.temp():
        memory.get_tensor((100,), torch.float32)
        with memory.temp():
            memory.get_tensor((50,), torch.int32)
    memory.get_tensor((10,), torch.float32)
    scratch = peak.end()
    assert scratch == 600
    assert memory.max_pointer == max(before, 256 + 600)
    # Ending twice is ending once.
    assert peak.end() == 600
    assert memory.max_pointer == max(before, 256 + 600)


def _scene(scene_cls):
    """Glass and a mirror floor: a splitting batch, with bounces."""
    from algan import (
        DOWN,
        GRAY_D,
        LEFT,
        ORIGIN,
        OUT,
        RIGHT,
        UP,
        WHITE,
        DirectionalLight,
        MeshLambertMaterial,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Off,
        Prism,
        Sphere,
    )

    with Off():
        scene_cls.clear_lights()
        DirectionalLight(
            location=RIGHT * 3 + UP * 4 + OUT * 5, target=ORIGIN, intensity=3.0
        ).spawn(animate=False)
        floor = Prism(width=8, height=0.2, depth=8, color=GRAY_D)
        floor.set_material(MeshStandardMaterial(metalness=0.6, roughness=0.05))
        floor.move(DOWN * 1.2).spawn(animate=False)
        glass = Sphere(radius=0.9).set_material(
            MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)
        )
        glass.move(LEFT * 0.6).spawn(animate=False)
        box = Prism(width=0.8, height=0.8, depth=0.8)
        box.set_material(MeshLambertMaterial(color=WHITE))
        box.move(RIGHT * 1.2 + DOWN * 0.6).spawn(animate=False)
    glass.move(RIGHT * 0.4)


def _render(disabled, budget):
    """The scene's frames, and how its tiles were planned and retried."""
    import pytest
    import torch

    from algan import SETTINGS, SMOKE_TEST, Scene, SceneManager

    seen = {"ratios": [], "estimates": []}
    plan = tracer._auto_primary_per_tile
    split = tracer._split_pool_ratio

    def recorded_plan(memory, ratio, *args, **kwargs):
        seen["ratios"].append(int(ratio))
        return plan(memory, ratio, *args, **kwargs)

    def recorded_split(*args, **kwargs):
        ratio = split(*args, **kwargs)
        seen["estimates"].append(int(ratio))
        return ratio

    for key in tracer._WAVEFRONT_TILE_STATS:
        tracer._WAVEFRONT_TILE_STATS[key] = 0
    SceneManager.reset()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
        mp.setattr(tracer, "_auto_primary_per_tile", recorded_plan)
        mp.setattr(tracer, "_split_pool_ratio", recorded_split)
        # These tiles are small; let them read as scratch rates.
        mp.setattr(tracer, "_SCRATCH_SAMPLE_PRIMARIES", 256)
        mp.setattr(SETTINGS.computing, "cpu_render_memory_budget", budget)
        try:
            video = SMOKE_TEST.set(resolution=(160, 120), frames_per_second=6)
            with Scene(video_settings=video) as scene:
                _scene(Scene)
                frames = [
                    b.clone() for b in scene.get_frames(0, _FRAMES, post_processes=())
                ]
        finally:
            SceneManager.reset()
    seen.update(tracer._WAVEFRONT_TILE_STATS)
    return torch.cat(frames, 0), seen


#: The arena the render gets (``cpu_render_memory_budget``): small enough
#: that the estimate's tiles run out of it, large enough for one frame. The
#: tiles get what the chunk leaves free, so a change to what a chunk holds
#: moves this: once the fragment stream stopped being held through the
#: resolve, 20 MB left the tiles 0.3 MB more, the estimate's tiles fit after
#: one halving, and both arms retried four times. 17, 18 and 19 MB each show
#: the record saving retries.
_BUDGET = 18 << 20


def _render_arm(arm, out_path):
    """One arm, in this (child) process; the frames go to ``out_path``."""
    from algan import SETTINGS

    SETTINGS.raytracing.set(samples_per_pixel=1, shadows=True, max_bounces=3)
    # The drain sorts its rays only from ``wf_ray_sort_min`` up, so whether a
    # bounce is sorted depends on how many rays its tile holds; a sorted
    # bounce takes a pixel's own rays in another order, which moves its float
    # sums and its glossy claim. That is tile size, not tile planning, so it
    # is pinned off here.
    SETTINGS.raytracing.experimental.set(wf_ray_sort=False)
    frames, seen = _render({"adaptivepool"} if arm == "fixed" else set(), _BUDGET)
    torch.save(frames, out_path)
    return seen


def _child(arm, out_path):
    """Render one arm in a child process of its own, on one CPU thread: the
    arms share nothing but the kernel cache on disk.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("ALGAN_OPT_DISABLE", "ALGAN_WAVEFRONT")):
            env.pop(name)
    env.update(
        ALGAN_RENDER_DEVICE="cpu",
        ALGAN_USE_DAEMON="0",
        ALGAN_PRECOMPILE_JOBS="0",
        PYTHONPATH=os.pathsep.join(
            p for p in (str(_REPO_ROOT), env.get("PYTHONPATH")) if p
        ),
    )
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arm, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_REPORT)]
    assert lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len(_REPORT) :])


def test_learned_tiles_render_the_same_image_with_fewer_retries(tmp_path):
    fixed = _child("fixed", tmp_path / "fixed.pt")
    learned = _child("learned", tmp_path / "learned.pt")
    fixed_frames = torch.load(tmp_path / "fixed.pt")
    learned_frames = torch.load(tmp_path / "learned.pt")
    report = {"fixed": fixed, "learned": learned}

    # The scene is what this is about, or it proves nothing: a splitting
    # batch whose estimate-sized tiles ran out of arena, over several chunks.
    estimate = max(fixed["estimates"])
    assert estimate > 1, report
    assert fixed["oom_retries"] > 0, report
    assert fixed["chunks"] > 1, report
    assert set(fixed["ratios"]) == {estimate}, report
    assert float(fixed_frames.float().std()) > 1.0
    # The later chunks planned from what the tiles used: a smaller ratio, and
    # fewer attempts thrown away for the arena.
    assert min(learned["ratios"]) < estimate, report
    assert learned["oom_retries"] < fixed["oom_retries"], report
    # The same chunks, so the same batch windows: an equal image is the
    # tiles' doing.
    assert learned["chunks"] == fixed["chunks"], report
    assert torch.equal(fixed_frames, learned_frames), int(
        (fixed_frames != learned_frames).sum()
    )


if __name__ == "__main__":
    # One CPU thread (module docstring); set before the render brings the
    # Taichi program up.
    import algan.rendering.taichi_runtime as _runtime

    _init_kwargs = _runtime.taichi_init_kwargs

    def _one_thread():
        kwargs = _init_kwargs()
        kwargs["cpu_max_num_threads"] = 1
        return kwargs

    _runtime.taichi_init_kwargs = _one_thread
    print(_REPORT + json.dumps(_render_arm(*sys.argv[1:3])))
