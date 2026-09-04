"""What the render loop lets a batch cost when the render device is MPS.

`DESIGN_mps_support.md` §1.4: `materials_and_lighting` dies on Metal after
roughly two thirds of its frames, silently, on both kernel compilers, having
logged the arena binary-searching twice. This script asks whether that is a
Metal fault at all -- and it can ask on a box with no Metal in it, because the
suspicion is about *batch sizing*, and batch sizing is device-branched Python.

The two budgets that bound what a frame window may cost **outside** the arena
are `RenderLoopMixin._gpu_merge_headroom_bytes` and `_render_device_prep_budget`.
Both used to branch on the device, return a real figure for `cuda` and for
`cpu`, and return `float("inf")` for everything else -- which is MPS. Every
guard they feed is a comparison against that figure:

    if require_estimates_fit and estimated_merge_peak > headroom:  # never true
    upper = _max_duration_that_fits(total_frames, project_fits)    # always all

So on Metal the window was bounded by the arena alone, while the merge and the
projection scratch -- which do not live in the arena -- were bounded by nothing.
`_render_device_pool_bytes` is the fix; these two arms are its A/B:

    native    what the engine does now, on whatever device this box has
    inf       the defect reinstated: both budgets forced back to infinity

Nothing else differs -- same scene, same 1536 MiB pool the gate pins, same
arena fraction -- and the `inf` arm restores the old branch rather than the
`native` arm computing a bound of its own, so no arithmetic in this file can
flatter the fix.

**And the answer is no: that defect is not §1.4.** Measured on Linux CPU, the
two arms take the *same* three windows -- 58, 47 and 74 frames -- and peak at
4.79 GB (`inf`) against 5.03 GB (`native`). The headroom the fix bounds is only
consulted by guards that `merge_on_gpu_active` / `project_on_gpu_active` gate
on a CUDA device, so on Metal almost nothing reads it; the fix closes a real
hole (nothing should hand a device an unsatisfiable budget) and closes nothing
that was killing this render.

What the same measurement *does* establish is the size of the thing: this scene
wants ~5 GB of host memory at PREVIEW behind a 1536 MiB pool, and it wants it in
the third window -- which is exactly where §1.4 saw Metal stop, on a runner with
7 GB shared between host and GPU. So `--max-window` is the discriminator, not
`--headroom`: a render that survives at 8 frames per window and dies at 74 died
of what a window costs.

Usage:
  uv run python benchmarks/_mps_batch_budget_repro.py --headroom inf
  uv run python benchmarks/_mps_batch_budget_repro.py --headroom native
  # the discriminator:
  uv run python benchmarks/_mps_batch_budget_repro.py --max-window 8
  # what a Mac user gets, with no gate pinning the pool around them:
  uv run python benchmarks/_mps_batch_budget_repro.py --no-pin-pool

`--ceiling-gb` is a watchdog, not a cap: `_memory_cap.py` refuses to bound a
real render (a failed commit segfaults inside native code rather than raising),
so this samples RSS and kills the process on the way past the ceiling, which
leaves the peak readable rather than taking the box down with it. Pass `0` to
disable it, which is what a Metal arm wants -- there the memory that matters is
not all the process's own to sample, and a watchdog would fire on the wrong
thing.
"""

from __future__ import annotations

import argparse
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
    """One line of whatever the render device will admit to holding."""
    import torch

    try:
        if torch.backends.mps.is_available():
            return (
                f" mps_alloc={torch.mps.current_allocated_memory() / 1e9:.2f}GB"
                f" mps_driver={torch.mps.driver_allocated_memory() / 1e9:.2f}GB"
            )
    except Exception:
        pass
    return ""


def start_rss_watchdog(ceiling_gb):
    """Sample RSS, remember the peak, and kill the process past ``ceiling_gb``."""
    ceiling = int(ceiling_gb * (1 << 30)) if ceiling_gb else 0

    def sample():
        while True:
            rss = _rss_bytes()
            PEAK["rss"] = max(PEAK["rss"], rss)
            if ceiling and rss > ceiling:
                print(
                    f"\nWATCHDOG: RSS {rss / 1e9:.2f} GB passed the "
                    f"{ceiling_gb} GB ceiling; killing the render.",
                    flush=True,
                )
                print(f"PEAK-RSS {PEAK['rss'] / 1e9:.2f} GB (ceiling hit)", flush=True)
                os._exit(42)
            time.sleep(0.1)

    threading.Thread(target=sample, daemon=True).start()


def install_probes(headroom_mode, pin_pool):
    """Trace every batch, and branch the two out-of-arena budgets by arm."""
    from algan.render_loop import RenderLoopMixin
    from algan.utils import memory_utils

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
        yield from real_batch(self, primitives, start_ind, end_ind, *args, **kwargs)

    RenderLoopMixin._render_primitive_batch = batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headroom", choices=("native", "inf"), default="native")
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
    install_probes(args.headroom, args.pin_pool)

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
        f"ARM headroom={args.headroom} scene={args.scene} "
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
