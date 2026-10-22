"""Ray and node-visit counts of a benchmark scene, per traversal switch arm.

Turns the traversal estimates into arithmetic: renders a workload scene once
per arm with the trace counters compiled in (``ALGAN_TRACE_COUNTERS``, see
``algan/rendering/raytracing/trace_counters_taichi.py``) and prints, per arm,
how many shadow rays were traced, how opaque-first answered them, and what
every class of walk spent in sibling-group (node) tests and primitive tests::

    python benchmarks/_trace_counters_report.py --scene graphics --quality UHD --seconds 0.5
    python benchmarks/_trace_counters_report.py --scene graphics --quality PREVIEW --seconds 6 --verify
    python benchmarks/_trace_counters_report.py --scene graphics --seconds 6 --at 3 --resolution 320x180

Arms are ``SETTINGS.raytracing.experimental.shadow_anyhit`` values, set
in-process between renders: the default compares the shipped ``auto`` with
the opt-in ``opaquefirst``. The node layout is import-time
(``ALGAN_BVH_NODE_WORDS=0`` for the original lanes), so it is one process per
layout. ``--verify`` also re-derives every opaque-first answer with the
ordered march and counts the rays that differ in any bit
(``shadow_opaque_first_mismatch``; 0 on the CPU, 56,927 on an RTX 5080 at
graphics UHD 0.5 s).

Counting kernels run slower (atomics, and their source keys are poisoned so
each pays the full frontend once): read counts from here and time with
``e2e_scene.py`` in separate processes. ``--json`` writes every arm's counts.
CPU or CUDA; the counters use 64-bit atomics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene", default="graphics", help="benchmarks/performance/<scene>_scene.py"
    )
    parser.add_argument("--quality", default="PREVIEW")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="authored clip length (the scene's default if unset)",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="override the preset's WxH (e.g. 320x180 for a CPU readout)",
    )
    parser.add_argument(
        "--arms",
        default="auto,opaquefirst",
        help="comma-separated SETTINGS.raytracing.experimental.shadow_anyhit "
        "values, one render each (node words are import-time: "
        "ALGAN_BVH_NODE_WORDS=0 in the environment for the lanes arm)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ALGAN_TRACE_COUNTERS=verify: count opaque-first mismatches",
    )
    parser.add_argument(
        "--at",
        default=None,
        help="comma-separated clip times: render these stills "
        "(save_frame) instead of the whole video",
    )
    parser.add_argument("--max-bounces", type=int, default=None)
    parser.add_argument("--json", default=None, help="write all arms' counts here")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="where the arm videos go (default: a temp directory)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    # Import-time switch: it has to be in the environment before algan loads.
    os.environ["ALGAN_TRACE_COUNTERS"] = "verify" if args.verify else "1"
    os.environ.setdefault("ALGAN_USE_DAEMON", "0")
    sys.path.insert(0, str(HERE / "performance"))

    import algan
    from algan import SETTINGS, Scene
    from algan.rendering.raytracing import trace_counters_taichi as tc
    from algan.rendering.raytracing import tracer
    from algan.scene_manager import SceneManager

    module = importlib.import_module(f"{args.scene}_scene")
    settings = getattr(algan, args.quality)
    if args.resolution:
        w, h = (int(v) for v in args.resolution.lower().split("x"))
        settings = settings.set(resolution=(w, h))
    seconds = args.seconds
    if seconds is None:
        import inspect

        seconds = inspect.signature(module.scene).parameters["seconds"].default
        if seconds is inspect.Parameter.empty:
            seconds = 1.0

    modes = []
    select = tracer._select_shadow_mode

    def recording(*a, **k):
        mode = select(*a, **k)
        modes.append(mode)
        return mode

    tracer._select_shadow_mode = recording

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(os.environ.get("TEMP", "/tmp")) / "algan_trace_counters"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in args.arms.split(","):
        SceneManager.reset()
        snapshot = SETTINGS.snapshot()
        try:
            module.scene(seconds)
            SETTINGS.raytracing.experimental.set(shadow_anyhit=arm)
            if args.max_bounces is not None:
                SETTINGS.raytracing.set(max_bounces=args.max_bounces)
            modes.clear()
            tc.reset()
            t0 = time.perf_counter()
            if args.at:
                Scene.save_frame(
                    str(out_dir / f"{args.scene}_{arm}.png"),
                    settings,
                    at=[float(t) for t in args.at.split(",")],
                    overwrite=True,
                )
            else:
                Scene.save_video(
                    str(out_dir / f"{args.scene}_{arm}.mp4"), settings, overwrite=True
                )
            elapsed = time.perf_counter() - t0
            counts = tc.read()
        finally:
            SETTINGS.restore(snapshot)
        results[arm] = {
            "counts": counts,
            "shadow_modes": sorted(set(modes)),
            "seconds": round(elapsed, 2),
        }
        print(
            f"\n[{args.scene} {args.quality} {settings.resolution[0]}x"
            f"{settings.resolution[1]}] arm {arm}: shadow modes "
            f"{sorted(set(modes))}, render {elapsed:.1f} s (counting)"
        )
        print(tc.format_counts(counts), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
