"""How the two calling conventions would trade off over a WHOLE render.

The per-launch A/B (``_arena_view_real_kernel_ab.py``) prices one launch: the
arena arm costs ~2% of device time and saves ~1.3 ms of HOST time. Which way
that nets out over a render depends on one number the per-launch harness cannot
see -- how much device work a single launch actually carries -- so this counts
the real launches and times the real enqueues in a real render.

Usage:
  .venv/Scripts/python.exe benchmarks/_arena_launch_budget.py W H SSAA
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

REC = {"n": 0, "host_s": 0.0, "threads": set(), "nd": 0, "each": []}
_LOCK = threading.Lock()


def install_hook():
    import torch

    import algan.rendering.raytracing.sheet_resolve_taichi as srt

    real = srt.sheet_resolve_shade

    def hook(*args):
        t0 = time.perf_counter()
        out = real(*args)
        dt = time.perf_counter() - t0
        with _LOCK:
            REC["n"] += 1
            REC["host_s"] += dt
            REC["threads"].add(threading.get_ident())
            REC["each"].append(dt)
            REC["nd"] = sum(1 for a in args if isinstance(a, torch.Tensor))
        return out

    srt.sheet_resolve_shade = hook


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    ssaa = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    from algan.utils.profiling_utils import enable_taichi_kernel_profiler
    enable_taichi_kernel_profiler()

    import taichi as ti
    from _arena_view_real_capture import build_scene  # noqa: E402

    import algan  # noqa: F401
    from algan import Scene

    install_hook()
    build_scene(width, height, ssaa)

    t0 = time.perf_counter()
    Scene.save_video(os.path.join("benchmarks", "_arena_budget_probe"),
                     overwrite=True)
    wall = time.perf_counter() - t0

    dev_ms = 0.0
    try:
        info = ti.profiler.query_kernel_profiler_info("sheet_resolve_shade")
        # ``counter`` counts OFFLOADED TASKS, not launches, and ``avg`` averages
        # over them -- their product is still the kernel's total device time.
        dev_ms = info.avg * info.counter
    except Exception as exc:  # noqa: BLE001
        print("profiler query failed:", exc)

    n = REC["n"]
    # The hook sits on the public name, so it counts what the CALL SITE passes.
    # Since the arena conversion the kernel itself takes far fewer -- that
    # count is on the launcher's spec, not here.
    import algan.rendering.raytracing.sheet_resolve_taichi as srt
    launcher = getattr(srt, "_sheet_resolve_shade_launch", None)
    bound = len(launcher.arena_spec) if launcher is not None else 0

    print()
    print(f"resolution {width}x{height} ssaa={ssaa} "
          f"(internal {width * ssaa}x{height * ssaa})")
    print(f"render wall                 {wall:8.2f} s")
    print(f"sheet_resolve_shade launches{n:8d}   "
          f"(threads {len(REC['threads'])}, {REC['nd']} ndarray args at the "
          f"call site, {REC['nd'] - bound + 5} at the kernel)")
    print(f"  device total              {dev_ms / 1000:8.3f} s   "
          f"per launch {dev_ms / max(n, 1):8.3f} ms")
    print(f"  host enqueue total        {REC['host_s']:8.3f} s   "
          f"per launch {REC['host_s'] / max(n, 1) * 1000:8.3f} ms")
    # Launch 1 carries the compile (cold) or the offline-cache load (warm);
    # only the ones after it are the steady-state enqueue cost.
    print("  host enqueue, each launch  "
          + " ".join(f"{d * 1000:.2f}" for d in REC["each"]) + " ms")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
