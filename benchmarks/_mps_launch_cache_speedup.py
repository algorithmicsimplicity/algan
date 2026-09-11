"""How much render wall time the Quadrants ndarray launch cache actually returns.

``_mps_launch_cache_check.py`` answers *is it correct and does it engage*. It
times two renders per arm, which cannot resolve a single-digit-percent effect on
this runner. This script answers *how much*, and it does so in two independent
ways in one process:

* **Budget.** One instrumented render per arm wraps the outermost
  ``Kernel.__call__`` and accumulates the time spent inside it. That is the
  whole host-side launch path -- the Python this change removes plus the C++
  enqueue it cannot -- so ``dispatch_seconds / wall_seconds`` is the ceiling on
  what any launch-path change can move, and the off/on difference in
  ``dispatch_seconds`` is what this one moves. Low noise: it sums hundreds of
  calls inside a single render.
* **End to end.** ``--repeats`` alternating warm renders per arm, with the
  instrumentation removed and telemetry and VERIFY off, reported as medians with
  a bootstrap CI on the relative difference. This is what a user would feel, and
  it is the noisy measurement -- the CI is the point of it.

Both arms keep zero-copy and its fences on; only ``fast.ENABLED`` moves. Every
render is warm (plans, kernels and the arena are warmed per arm first), and the
arms alternate in both orders so a thermal drift cannot be read as a win.

    .venv/bin/python benchmarks/_mps_launch_cache_speedup.py
    .venv/bin/python benchmarks/_mps_launch_cache_speedup.py --scenes fast,materials_and_lighting --repeats 12

Needs ``latex=true`` on the harness for the full-render scenes. Emits one
``SPEEDUP_SCENE`` JSON line per scene plus a final ``SPEEDUP_SUMMARY``.
"""

# Runtime kernel annotations; do not postpone them.
import argparse
import json
import os
import random
import runpy
import statistics
import sys
import time
from contextlib import chdir
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from algan import LD, OUT, RIGHT, UP, Off, PointLight, Scene, Sphere, Square
from algan.rendering.taichi_runtime import init_taichi
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.taichi_compat import BACKEND, submodule, ti
from algan.utils import taichi_fast_launch as fast


def sync():
    if submodule("lang.impl").get_runtime()._prog is not None:
        ti.sync()
    device = render_device().type
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class DispatchMeter:
    """Accumulate wall time spent inside the outermost ``Kernel.__call__``.

    Installed *over* whatever chain is already on the class (arch guard ->
    zero-copy -> fast dispatcher), so it sees every launch exactly once and
    measures the same span in both arms. Uninstalled before any timed render:
    a ``perf_counter`` pair per launch is itself a few hundred nanoseconds,
    which is real next to the ~20 us this change is worth.
    """

    def __init__(self):
        self.kernel_cls = submodule("lang.kernel").Kernel
        self.previous = None
        self.seconds = 0.0
        self.launches = 0

    def __enter__(self):
        previous = self.kernel_cls.__call__
        self.previous = previous
        meter = self

        def metered(kernel, *args, **kwargs):
            started = time.perf_counter()
            try:
                return previous(kernel, *args, **kwargs)
            finally:
                meter.seconds += time.perf_counter() - started
                meter.launches += 1

        self.kernel_cls.__call__ = metered
        return self

    def __exit__(self, *exc):
        self.kernel_cls.__call__ = self.previous
        self.previous = None
        return False


def bootstrap_relative_ci(off, on, draws=20000, seed=20260911):
    """95% CI on ``(median(off) - median(on)) / median(off)``, as a percentage.

    Resampled per arm, so the interval reflects the run-to-run spread these
    samples actually showed rather than an assumed distribution. A CI that
    straddles zero means this experiment did not separate the arms.
    """
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        a = statistics.median(rng.choices(off, k=len(off)))
        b = statistics.median(rng.choices(on, k=len(on)))
        ratios.append((a - b) / a * 100.0)
    ratios.sort()
    return ratios[int(0.025 * draws)], ratios[int(0.975 * draws)]


def load_scene(path):
    if path is None:
        with Off():
            PointLight(location=UP * 3 + OUT * 4).spawn()
            Square(size=1.8).spawn()
            Sphere(radius=0.65).move(RIGHT * 1.1 + UP * 0.35).spawn()
            Sphere(radius=0.5).move(OUT * 1.2 + RIGHT * 0.3).spawn()
        Scene.wait(1)
    else:
        runpy.run_path(str(path), run_name="_algan_launch_cache_scene")


def measure_scene(name, path, output, repeats, frames):
    scene_root = ROOT if path is None else path.parent
    if scene_root.name == "scenes":
        scene_root = scene_root.parent
    snapshot = SETTINGS.snapshot()
    was_enabled, was_verify = fast.ENABLED, fast.VERIFY
    fast.VERIFY = False
    fast.set_telemetry_enabled(False)
    try:
        arena_mib = 512 if render_device().type == "cpu" else 1536
        SETTINGS.computing.set(
            available_memory_override=arena_mib * 1024**2, torch_compile=False
        )
        SETTINGS.paths.set(cache_directory=str(output / "cache"))
        with chdir(scene_root), Scene() as scene:
            load_scene(path)
            SETTINGS.raytracing.set(denoise=False)
            duration = max(float(scene._recorded_end_time_for_render()), 0.1)
            # More frames per measurement than the correctness probe takes:
            # launch count scales with them while the per-call setup of
            # save_frame does not, so the quantity under test carries more of
            # each sample.
            times = [duration * (i + 0.5) / frames for i in range(frames)]
            quality = LD.set(resolution=(320, 180), frames_per_second=10)
            if path is not None and "path_traced" in path.parts:
                quality = quality.set(resolution=(128, 72), frames_per_second=5)

            def render(label, enabled):
                fast.set_enabled(enabled)
                sync()
                started = time.perf_counter()
                scene.save_frame(
                    str(output / f"{name}-{label}.png"),
                    video_settings=quality,
                    at=times,
                )
                sync()
                return time.perf_counter() - started

            # Warm each arm's plans, kernels and arena before anything counts.
            render("warm-off", False)
            render("warm-on", True)
            render("warm-off2", False)
            render("warm-on2", True)

            budget = {}
            for label, enabled in (("off", False), ("on", True)):
                with DispatchMeter() as meter:
                    wall = render(f"metered-{label}", enabled)
                budget[label] = {
                    "wall_seconds": wall,
                    "dispatch_seconds": meter.seconds,
                    "launches": meter.launches,
                    "dispatch_share_pct": meter.seconds / wall * 100.0,
                    "us_per_launch": meter.seconds / max(meter.launches, 1) * 1e6,
                }

            samples = {False: [], True: []}
            for index in range(repeats):
                # Alternate the within-pair order so a monotonic drift across
                # the run cancels instead of landing on one arm.
                order = (False, True) if index % 2 == 0 else (True, False)
                for enabled in order:
                    samples[enabled].append(
                        render(f"timed-{index}-{int(enabled)}", enabled)
                    )

            off, on = samples[False], samples[True]
            low, high = bootstrap_relative_ci(off, on)
            result = {
                "scene": name,
                "frames": frames,
                "resolution": list(quality.resolution),
                "repeats": repeats,
                "budget": budget,
                "dispatch_saved_seconds": (
                    budget["off"]["dispatch_seconds"] - budget["on"]["dispatch_seconds"]
                ),
                "dispatch_saved_us_per_launch": (
                    budget["off"]["us_per_launch"] - budget["on"]["us_per_launch"]
                ),
                "off_median_seconds": statistics.median(off),
                "on_median_seconds": statistics.median(on),
                "off_min_seconds": min(off),
                "on_min_seconds": min(on),
                "relative_gain_pct": (
                    (statistics.median(off) - statistics.median(on))
                    / statistics.median(off)
                    * 100.0
                ),
                "relative_gain_ci95_pct": [low, high],
                "off_samples_seconds": off,
                "on_samples_seconds": on,
            }
            print("SPEEDUP_SCENE " + json.dumps(result), flush=True)
            return result
    finally:
        fast.set_enabled(was_enabled)
        fast.VERIFY = was_verify
        SETTINGS.restore(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        default="fast,materials_and_lighting,lit_and_shadowed",
        help="smoke, all, or comma-separated scene stems",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "algan_outputs" / "launch-speedup"
    )
    args = parser.parse_args()
    if not 2 <= args.repeats <= 100:
        parser.error("repeats must be between 2 and 100")
    if not 1 <= args.frames <= 60:
        parser.error("frames must be between 1 and 60")
    assert BACKEND == "quadrants", "this measures the Quadrants dispatcher"
    assert fast.skipped_reason() is None, fast.skipped_reason()
    init_taichi()
    device = render_device().type
    arch = submodule("lang.impl").get_runtime()._arch
    if os.environ.get("ALGAN_RENDER_DEVICE") == "mps":
        assert device == "mps", "the MPS arm resolved the wrong device"
        assert arch == ti.metal, "the MPS arm resolved the wrong compiler arch"
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "SPEEDUP_METADATA "
        + json.dumps({"device": device, "arch": str(arch), "torch": torch.__version__}),
        flush=True,
    )

    available = {"smoke": None, "fast": ROOT / "tests/fast/scene.py"}
    for suite in ("full_renders", "path_traced"):
        available.update(
            {p.stem: p for p in sorted((ROOT / f"tests/{suite}/scenes").glob("*.py"))}
        )
    chosen = list(available) if args.scenes == "all" else args.scenes.split(",")
    unknown = set(chosen) - available.keys()
    if unknown:
        parser.error(f"unknown scenes: {sorted(unknown)}")
    if chosen != ["smoke"]:
        import manimpango

        for font in (ROOT / "tests/assets/fonts").glob("*.ttf"):
            if not manimpango.register_font(str(font)):
                raise RuntimeError(f"could not register {font}")

    results = []
    for name in chosen:
        results.append(
            measure_scene(name, available[name], args.output, args.repeats, args.frames)
        )
        (args.output / "speedup.json").write_text(json.dumps(results, indent=2))

    dispatch = sum(r["budget"]["off"]["dispatch_seconds"] for r in results)
    wall = sum(r["budget"]["off"]["wall_seconds"] for r in results)
    saved = sum(r["dispatch_saved_seconds"] for r in results)
    print(
        "SPEEDUP_SUMMARY "
        + json.dumps(
            {
                "scenes": [r["scene"] for r in results],
                "dispatch_share_of_wall_pct": dispatch / wall * 100.0,
                "dispatch_saved_pct_of_wall": saved / wall * 100.0,
                "per_scene_gain_pct": {
                    r["scene"]: [r["relative_gain_pct"], *r["relative_gain_ci95_pct"]]
                    for r in results
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
