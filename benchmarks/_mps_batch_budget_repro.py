"""What a render holds on, and where it holds it, when the device is MPS.

`DESIGN_mps_support.md` §1.4: `materials_and_lighting` dies on Metal after
roughly two thirds of its frames, silently, on both kernel compilers. This
script found out why, and it is a **leak on the device**, not a batch that is
too big:

    batch 0:8      mps_alloc=0.64GB  mps_driver=0.68GB
    batch 56:64    mps_alloc=2.28GB  mps_driver=3.54GB
    batch 96:104   mps_alloc=4.59GB  mps_driver=6.12GB
    batch 120:128  mps_alloc=6.74GB  mps_driver=8.27GB   <- Trace/BPT trap: 5

`mps_alloc` is `torch.mps.current_allocated_memory()`: bytes torch considers
**live**. It rises monotonically and never falls, past the 7 GB the runner has,
on a machine `sysctl` reports as having no swap at all. Host RSS stayed under
1.2 GB throughout, which is why every host-side measurement here missed it.

The leak is `rendering/mps_zero_copy.py`'s import cache. It holds a torch
storage per buffer it has handed a kernel -- it must, since Taichi's imported
ndarray keeps no reference of its own -- and it was released once, at the end
of the render job. The arena is one storage the job reuses, so that looked
sufficient; every *other* kernel argument is a fresh allocation on every batch,
and each was pinned until the last frame. `release_torch_memory` now clears it,
which is also what makes `torch.mps.empty_cache()` mean anything on Metal.

The arms:

    (default)      the engine as it stands
    --leak-cache   the defect reinstated: the cache never released mid-render
    --headroom inf a second, unrelated defect, kept for the record (below)
    --max-window N cap every frame window, which is NOT the lever (below)

Read `pinned=` in the batch lines against `mps_alloc=`: that is the cache's
share of what torch holds live, and the two moving together is the claim.

**Two measured negatives, kept because they cost a day between them.**
`--headroom inf` restores the `float("inf")` that `_gpu_merge_headroom_bytes`
and `_render_device_prep_budget` used to return for any device that was not
CUDA or CPU. That is a real hole -- an infinite budget makes every guard it
feeds unsatisfiable -- and `_render_device_pool_bytes` closes it, but it is not
§1.4: on Linux CPU both arms take the same three windows (58, 47, 74 frames)
and peak at 4.79 GB against 5.03 GB. And `--max-window 8`, which looked like
the discriminator, dies on Metal at the same *frame* as an unbounded window,
because a leak proportional to frames rendered does not care how they were
grouped.

Usage:
  uv run python benchmarks/_mps_batch_budget_repro.py --leak-cache
  uv run python benchmarks/_mps_batch_budget_repro.py
  # what a Mac user gets, with no gate pinning the pool around them:
  uv run python benchmarks/_mps_batch_budget_repro.py --no-pin-pool

`--ceiling-gb` is a watchdog, not a cap: `_memory_cap.py` refuses to bound a
real render (a failed commit segfaults inside native code rather than raising),
so this samples RSS and kills the process on the way past the ceiling, which
leaves the peak readable rather than taking the box down with it. Pass `0` to
disable it, which is what a Metal arm wants -- there the memory that matters is
not the process's own to sample, and a watchdog would fire on the wrong thing.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import resource
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set before `import algan`: a warm daemon would render this against whatever
# ran before it, and the point here is a batch schedule.
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

MIB = 1 << 20
#: `scripts/gate/backend_pixel_ab.py`'s figure, which is `test_full_renders.py`'s.
#: It replaces the free-memory measurement so a window does not move with the box.
POOL_BYTES = 1536 * MIB

PEAK = {"rss": 0}


def _rss_bytes():
    """Resident bytes of this process, on either box."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:  # Linux without psutil.
        with open("/proc/self/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # ru_maxrss is the high-water mark, not the current figure, and its unit
    # differs by platform: bytes on macOS, kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _device_memory_note():
    """One line of whatever the render device will admit to holding.

    ``mps_alloc`` is the one that mattered: torch's *live* MPS bytes, which on
    a healthy render rise and fall with a batch and which §1.4 found rising
    monotonically to 6.74 GB on a 7 GB machine. ``pinned`` is the import
    cache's share of it, so the two can be compared rather than correlated by
    eye.
    """
    import torch

    try:
        if not torch.backends.mps.is_available():
            return ""
        note = (
            f" mps_alloc={torch.mps.current_allocated_memory() / 1e9:.2f}GB"
            f" mps_driver={torch.mps.driver_allocated_memory() / 1e9:.2f}GB"
        )
        from algan.rendering.mps_zero_copy import cache_stats

        entries, storages, pinned = cache_stats()
        return f"{note} pinned={pinned / 1e9:.2f}GB({entries}e/{storages}s)"
    except Exception:
        return ""


#: Name of the render phase the sampler should credit what it sees to, and the
#: worst RSS seen inside each. A phase's *peak* is the number that matters --
#: the render's high-water mark decides whether the box survives it, and a
#: before/after delta around a phase that allocates and releases shows nothing.
PHASE = {"name": "startup"}
PHASE_PEAK: dict[str, int] = {}


@contextlib.contextmanager
def phase(name):
    """Credit the sampler's readings to ``name`` for the duration."""
    previous = PHASE["name"]
    PHASE["name"] = name
    try:
        yield
    finally:
        PHASE["name"] = previous


def start_rss_watchdog(ceiling_gb):
    """Sample RSS, remember the peak, and kill the process past ``ceiling_gb``."""
    ceiling = int(ceiling_gb * (1 << 30)) if ceiling_gb else 0

    def sample():
        while True:
            rss = _rss_bytes()
            PEAK["rss"] = max(PEAK["rss"], rss)
            name = PHASE["name"]
            if rss > PHASE_PEAK.get(name, 0):
                PHASE_PEAK[name] = rss
            if ceiling and rss > ceiling:
                print(
                    f"\nWATCHDOG: RSS {rss / 1e9:.2f} GB passed the "
                    f"{ceiling_gb} GB ceiling; killing the render.",
                    flush=True,
                )
                print(f"PEAK-RSS {PEAK['rss'] / 1e9:.2f} GB (ceiling hit)", flush=True)
                report_phases()
                os._exit(42)
            time.sleep(0.1)

    threading.Thread(target=sample, daemon=True).start()


def report_phases():
    """Print the worst RSS each phase was inside, worst first."""
    if not PHASE_PEAK:
        return
    print("\nPHASE-PEAKS (worst RSS observed while inside each):", flush=True)
    for name, peak in sorted(PHASE_PEAK.items(), key=lambda kv: -kv[1]):
        print(f"  {peak / 1e9:6.2f} GB  {name}", flush=True)


def install_probes(headroom_mode, pin_pool, leak_cache=False):
    """Trace every batch, and reinstate whichever defect this arm is about."""
    from algan.render_loop import RenderLoopMixin
    from algan.utils import memory_utils

    if leak_cache:
        # The §1.4 defect reinstated: the zero-copy import cache never released
        # while the render runs. Patched on the module rather than
        # reimplemented here, and both callers import the name inside the
        # function, so this arm is the engine minus exactly one behaviour.
        # It neutralises `taichi_runtime`'s end-of-job clear too, which is
        # faithful rather than sloppy: that clear ran after the last frame, so
        # over one render job it never freed a byte the render could use, and
        # this arm renders one job.
        from algan.rendering import mps_zero_copy

        mps_zero_copy.clear_import_cache = lambda: None

    # Only the CPU needs this: `available_memory_override` already sizes the
    # arena on a measured device, but the CPU branch reads `max_cpu_memory_used`
    # instead and would give the arms a different arena from the Mac's.
    if pin_pool and headroom_mode != "native":
        real_init = memory_utils.ManualMemory.__init__

        def init(self, portion, device=None, managed=True, *, num_bytes=None):
            # What `get_num_available_bytes` returns on MPS under the gate's
            # `available_memory_override`: the override itself, rather than the
            # CPU's `max_cpu_memory_used`. Sizing the arena from one pool on
            # both arms is what makes their frame windows comparable.
            if managed and num_bytes is None:
                num_bytes = int(POOL_BYTES * portion)
            real_init(
                self, portion, device=device, managed=managed, num_bytes=num_bytes
            )

        memory_utils.ManualMemory.__init__ = init

    if headroom_mode == "inf":
        # The defect reinstated, so the two arms differ in exactly the thing
        # under test. Restoring it here rather than keeping a bounded arm's
        # own arithmetic is what stops the comparison drifting away from the
        # engine: the `native` arm is whatever the engine currently does, and
        # nothing in this file can flatter it.
        def headroom(self):
            return float("inf")

        def prep_budget(self):
            return float("inf")

        RenderLoopMixin._gpu_merge_headroom_bytes = headroom
        RenderLoopMixin._render_device_prep_budget = prep_budget

    real_batch = RenderLoopMixin._render_primitive_batch

    def batch(self, primitives, start_ind, end_ind, *args, **kwargs):
        print(
            f"batch frames {start_ind}:{end_ind} ({end_ind - start_ind}f) "
            f"arena={len(self.memory) / 1e6:.0f}MB "
            f"used={self.memory.get_percent_used() * 100:.0f}% "
            f"headroom={self._gpu_merge_headroom_bytes() / 1e6:.0f}MB "
            f"rss={_rss_bytes() / 1e9:.2f}GB{_device_memory_note()}",
            flush=True,
        )
        with phase("render"):
            yield from real_batch(self, primitives, start_ind, end_ind, *args, **kwargs)

    RenderLoopMixin._render_primitive_batch = batch

    # The phases a batch passes through, in the order it passes through them.
    # The names are the methods' own, because the point of the reading is to
    # send someone to the code that holds the memory.
    for name in (
        "_get_batch_of_primitives",
        "_prewarm_render_batch",
        "_prepare_merged_host_scene",
    ):
        real = getattr(RenderLoopMixin, name)

        def wrap(real=real, name=name):
            def wrapped(self, *args, **kwargs):
                with phase(name):
                    return real(self, *args, **kwargs)

            return wrapped

        setattr(RenderLoopMixin, name, wrap())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headroom", choices=("native", "inf"), default="native")
    parser.add_argument(
        "--leak-cache",
        action="store_true",
        help=(
            "reinstate the §1.4 defect: never release the MPS zero-copy import "
            "cache while the render runs. The arm that dies"
        ),
    )
    parser.add_argument("--scene", default="materials_and_lighting")
    parser.add_argument(
        "--ceiling-gb",
        type=float,
        default=9.0,
        help="kill the process past this RSS; 0 disables the watchdog",
    )
    parser.add_argument(
        "--pin-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "pin available_memory_override to the gate's 1536 MiB (default). "
            "--no-pin-pool leaves the pool to the device's own measurement, "
            "which is what a Mac user with no gate around them actually gets"
        ),
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help=(
            "cap every fetched frame window at this many frames "
            "(SETTINGS.computing.max_animation_batch_size). The discriminator: "
            "a render that survives at 8 frames and dies at 74 died of what a "
            "window costs, and a render that dies at both did not"
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    start_rss_watchdog(args.ceiling_gb)
    install_probes(args.headroom, args.pin_pool, leak_cache=args.leak_cache)

    import importlib.util
    from pathlib import Path

    from algan import PREVIEW, SETTINGS, Scene
    from algan.scene_manager import SceneManager
    from algan.settings import _startup
    from algan.taichi_compat import describe_backend

    repo_root = Path(__file__).resolve().parents[1]
    scenes_dir = repo_root / "tests" / "full_renders" / "scenes"
    out_dir = Path(args.out or f"/tmp/mps-budget-{args.headroom}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # The suite's own working directory, so scene assets resolve by name.
    os.chdir(scenes_dir.parent)
    SETTINGS.paths.set(
        output_root=str(out_dir),
        output_directory=".",
        cache_directory=str(repo_root / "tests" / "full_renders" / "algan_cache"),
    )
    if args.pin_pool:
        SETTINGS.computing.set(available_memory_override=POOL_BYTES)
    if args.max_window:
        SETTINGS.computing.set(max_animation_batch_size=args.max_window)
    SceneManager.reset()

    print(
        f"ARM headroom={args.headroom} cache={'LEAKED' if args.leak_cache else 'released'} "
        f"scene={args.scene} "
        f"pool={'pinned-1536MiB' if args.pin_pool else 'measured'} "
        f"window<={args.max_window or 'unbounded'} "
        f"device={_startup.render_device().type} backend={describe_backend()}",
        flush=True,
    )
    started = time.perf_counter()
    scene_path = scenes_dir / f"{args.scene}.py"
    spec = importlib.util.spec_from_file_location("_algan_budget_scene", scene_path)
    module = importlib.util.module_from_spec(spec)
    with Scene() as scene:
        spec.loader.exec_module(module)
        scene.save_video(
            out_dir / f"{args.scene}.mp4",
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )
    elapsed = time.perf_counter() - started
    print(f"\nARM-DONE headroom={args.headroom} in {elapsed:.1f}s", flush=True)
    print(f"PEAK-RSS {PEAK['rss'] / 1e9:.2f} GB", flush=True)
    report_phases()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
