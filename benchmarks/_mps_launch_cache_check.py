"""Same-process launch/cache/parity validation for the Mac runner (also CPU/CUDA).

Run with the locked published Quadrants wheel, ALGAN_VIDEO_ENCODER=software:
    python benchmarks/_mps_launch_cache_check.py --scenes smoke
    python benchmarks/_mps_launch_cache_check.py --scenes all --iterations 200

"all" samples two times in the fast, six full-render and four path-traced
scenes; it is NOT a replacement for every frame in the baseline suites. It
needs the harness's latex=true. PNGs are compared before any lossy encoding.
Each case prints its result immediately and updates an artifact JSON. Timings
exclude VERIFY and detailed telemetry. The launcher switch never disables
zero-copy or its fences. No compiler rebuild or scalar-value cache is used.
"""

# Runtime kernel annotations; do not postpone them.
import argparse
import json
import os
import platform
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

import numpy as np
import torch
from PIL import Image

from algan import LD, OUT, RIGHT, UP, Off, PointLight, Scene, Sphere, Square
from algan.rendering import mps_zero_copy
from algan.rendering.taichi_runtime import init_taichi
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.taichi_compat import BACKEND, backend_version, submodule, ti
from algan.utils import taichi_fast_launch as fast


@ti.kernel
def probe(
    a0: ti.types.ndarray(),
    a1: ti.types.ndarray(),
    a2: ti.types.ndarray(),
    a3: ti.types.ndarray(),
    a4: ti.types.ndarray(),
    a5: ti.types.ndarray(),
    a6: ti.types.ndarray(),
    a7: ti.types.ndarray(),
    a8: ti.types.ndarray(),
    a9: ti.types.ndarray(),
    a10: ti.types.ndarray(),
    a11: ti.types.ndarray(),
    a12: ti.types.ndarray(),
    a13: ti.types.ndarray(),
    a14: ti.types.ndarray(),
    a15: ti.types.ndarray(),
    a16: ti.types.ndarray(),
    a17: ti.types.ndarray(),
    a18: ti.types.ndarray(),
    a19: ti.types.ndarray(),
    n: ti.i32,
    scale: ti.f32,
):
    a0[0] = (
        a1[0]
        + a2[0]
        + a3[0]
        + a4[0]
        + a5[0]
        + a6[0]
        + a7[0]
        + a8[0]
        + a9[0]
        + a10[0]
        + a11[0]
        + a12[0]
        + a13[0]
        + a14[0]
        + a15[0]
        + a16[0]
        + a17[0]
        + a18[0]
        + a19[0]
    ) * scale + n


def sync():
    # A completed render can reclaim its Program under host-memory pressure.
    # Such a reset has already drained outstanding work; do not sync a dead Program.
    if submodule("lang.impl").get_runtime()._prog is not None:
        ti.sync()
    device = render_device().type
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def launch_benchmark(iterations):
    # Fixed tiny allocations, independent of the iteration count. On Metal
    # each argument is a distinct nonzero-offset slice of the same MTLBuffer.
    storage = torch.ones(20 * 32, device=render_device())
    tensors = list(storage.split(32))
    native = []
    for tensor in tensors:
        if render_device().type == "mps":
            nd = mps_zero_copy.import_tensor(tensor)
            assert nd is not None
        else:
            nd = ti.ndarray(ti.f32, shape=32)
            nd.from_numpy(np.ones(32, dtype=np.float32))
        native.append(nd)
    sync()
    results = {}
    for name, arrays in (
        ("native_ndarray_dispatch", native),
        ("torch_wrapper_end_to_end", tensors),
    ):
        fast.set_enabled(True)
        probe(*arrays, 1000, 0.5)
        probe(*arrays, 1001, 1.0)
        fast.set_telemetry_enabled(False)
        fast.VERIFY = False
        samples = {False: [], True: []}
        complete = {False: [], True: []}
        for enabled in (False, True, True, False, False, True):
            fast.set_enabled(enabled)
            sync()
            started = time.perf_counter()
            for i in range(iterations):
                probe(*arrays, 1000 + i, 0.5 + (i % 2) * 0.5)
            enqueued = time.perf_counter()
            sync()
            ended = time.perf_counter()
            # Imported Metal ndarrays deliberately expose no host-copy API;
            # read their owning torch view, after the same device fences.
            if render_device().type == "mps":
                actual = tensors[0].cpu().numpy()
            elif isinstance(arrays[0], torch.Tensor):
                actual = arrays[0].cpu().numpy()
            else:
                actual = arrays[0].to_numpy()
            expected = 19 * (0.5 + ((iterations - 1) % 2) * 0.5) + 1000 + iterations - 1
            assert actual[0] == expected
            samples[enabled].append((enqueued - started) * 1e6 / iterations)
            complete[enabled].append((ended - started) * 1e6 / iterations)
        results[name] = {
            "off_enqueue_us": statistics.median(samples[False]),
            "on_enqueue_us": statistics.median(samples[True]),
            "off_complete_us": statistics.median(complete[False]),
            "on_complete_us": statistics.median(complete[True]),
            "enqueue_samples_us": {str(k): v for k, v in samples.items()},
        }
        print(
            "LAUNCH_CACHE_MICRO " + json.dumps({"case": name, **results[name]}),
            flush=True,
        )
    return results


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


def pixels(paths):
    return [np.asarray(Image.open(path)).copy() for path in paths]


def max_difference(left, right):
    assert len(left) == len(right) == 2
    assert all(a.shape == b.shape and a.size for a, b in zip(left, right))
    return max(
        int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
        for a, b in zip(left, right)
    )


def check_scene(name, path, output):
    output = output.resolve()
    scene_root = ROOT if path is None else path.parent
    if scene_root.name == "scenes":
        scene_root = scene_root.parent
    snapshot = SETTINGS.snapshot()
    old_enabled, old_verify = fast.ENABLED, fast.VERIFY
    try:
        arena_mib = 512 if render_device().type == "cpu" else 1536
        SETTINGS.computing.set(
            available_memory_override=arena_mib * 1024**2, torch_compile=False
        )
        SETTINGS.paths.set(cache_directory=str(output / "cache"))
        # Match each suite's harness: media paths are relative to the suite,
        # not to this benchmark's directory or the repository root.
        with chdir(scene_root), Scene() as scene:
            load_scene(path)
            SETTINGS.raytracing.set(denoise=False)
            duration = max(float(scene._recorded_end_time_for_render()), 0.1)
            times = [duration * 0.3, duration * 0.7]
            quality = LD.set(resolution=(320, 180), frames_per_second=10)
            if path is not None and "path_traced" in path.parts:
                quality = quality.set(resolution=(128, 72), frames_per_second=5)

            def render(label, enabled, verify=False, telemetry=False):
                fast.set_enabled(enabled)
                fast.VERIFY = verify
                fast.set_telemetry_enabled(telemetry)
                if telemetry:
                    fast.launch_report(reset=True)
                before = dict(mps_zero_copy.STATS)
                mps_zero_copy.LEFT_ON_THE_BUS.clear()
                sync()
                start = time.perf_counter()
                rendered = scene.save_frame(
                    str(output / f"{name}-{label}.png"),
                    video_settings=quality,
                    at=times,
                )
                sync()
                elapsed = time.perf_counter() - start
                if path is not None and "path_traced" in path.parts:
                    assert all(
                        item.render_plan.backend == "path_tracer" for item in rendered
                    ), "a path-traced scene silently selected a different renderer"
                result = pixels([item.output_path for item in rendered])
                bus = {k: mps_zero_copy.STATS[k] - before[k] for k in before}
                left = sorted(mps_zero_copy.LEFT_ON_THE_BUS)
                return result, elapsed, bus, left

            ref, _, _, _ = render("warm-off", False)
            ref2, _, _, _ = render("reference", False)
            warm, _, _, _ = render("warm-on", True, verify=True)
            actual, _, bus, left = render("verified", True, verify=True, telemetry=True)
            table = fast.launch_report()
            fast.set_telemetry_enabled(False)
            differences = {
                "off_repeat": max_difference(ref, ref2),
                "on_warm": max_difference(ref2, warm),
                "on_verified": max_difference(ref2, actual),
            }
            timings = {False: [], True: []}
            for index, enabled in enumerate((True, False, False, True)):
                _, elapsed, _, _ = render(f"timed-{index}", enabled)
                timings[enabled].append(elapsed)
            result = {
                "scene": name,
                "times": times,
                "resolution": quality.resolution,
                "max_channel_delta": differences,
                "launches": table,
                "zero_copy": bus,
                "left_on_the_bus": left,
                "warm_off_seconds": statistics.median(timings[False]),
                "warm_on_seconds": statistics.median(timings[True]),
                "warm_samples_seconds": {str(k): v for k, v in timings.items()},
            }
            totals = table["totals"]
            result["passed"] = (
                max(differences.values()) <= 1
                and totals["fast"] > 0
                and totals["cold"] == 0
                and totals["fallback"] == 0
                and totals["error"] == 0
                and bus["staged_arguments"] == 0
            )
            print("LAUNCH_CACHE_SCENE " + json.dumps(result), flush=True)
            return result
    finally:
        fast.set_telemetry_enabled(False)
        fast.set_enabled(old_enabled)
        fast.VERIFY = old_verify
        SETTINGS.restore(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes", default="smoke", help="smoke, all, or comma-separated scene stems"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "algan_outputs" / "launch-cache"
    )
    args = parser.parse_args()
    if not 1 <= args.iterations <= 5000:
        parser.error("iterations must be between 1 and 5000")
    assert BACKEND == "quadrants" and fast.skipped_reason() is None
    init_taichi()
    device = render_device().type
    arch = submodule("lang.impl").get_runtime()._arch
    if os.environ.get("ALGAN_RENDER_DEVICE") == "mps":
        assert device == "mps" and arch == ti.metal, (
            "the MPS arm resolved the wrong device"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "device": device,
        "arch": str(arch),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "compiler": backend_version(),
    }
    print("LAUNCH_CACHE_METADATA " + json.dumps(metadata), flush=True)
    results = {
        "metadata": metadata,
        "micro": launch_benchmark(args.iterations),
        "scenes": [],
    }
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
    for name in chosen:
        results["scenes"].append(check_scene(name, available[name], args.output))
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
    return 0 if all(r["passed"] for r in results["scenes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
