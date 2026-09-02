"""Warm in-process A/B of ``SETTINGS.computing.torch_compile``, with parity.

Renders one scene repeatedly in one process, alternating the compile switch
off and on, and reports the warm wall time of each arm plus the pixel
difference between them. In-process and alternating, because cross-process
wall-clock is noisy (``agent_guidance/memory_perf.md``); warm, because the
first render of each arm pays a one-off cost this benchmark is not about
(Taichi's kernel specialisation for the eager arm, Dynamo/Inductor's compile
for the compiled one) -- both are reported separately as "cold".

Usage::

    uv run python benchmarks/_torch_compile_ab.py [--scene tests/fast/scene.py]
        [--runs 3] [--quality PREVIEW] [--json out.json]

The scene file only *records* an animation, exactly like ``tests/fast/scene.py``;
this harness owns the Scene, the settings and the comparison. It is rebuilt for
every run so no run inherits another's timeline. The two arms are compared
frame by frame on lossless video, against the same tolerance the render suites
use (``tests/conftest.py``, ``MAX_CHANNEL_DIFFERENCE``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

# A warm daemon would carry state between arms, and this is a measurement.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_PROGRESS", "none")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "tests" / "fast" / "scene.py"
FONT_DIR = ROOT / "tests" / "assets" / "fonts"
#: The render suites' tolerance (tests/conftest.py MAX_CHANNEL_DIFFERENCE).
MAX_CHANNEL_DIFFERENCE = 2


def _register_test_fonts():
    """The fast scene pins ``Algan Test Sans``; make the vendored faces visible."""
    try:
        import manimpango
    except ImportError:
        return
    for face in sorted(FONT_DIR.glob("*.ttf")):
        manimpango.register_font(str(face))


def _load_scene(scene_file):
    spec = importlib.util.spec_from_file_location("_algan_ab_scene", scene_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_algan_ab_scene", None)


def _compare_videos(a_path, b_path):
    """``(max channel difference, worst frame, frame count)`` between two videos."""
    import cv2
    import numpy as np

    a = cv2.VideoCapture(str(a_path))
    b = cv2.VideoCapture(str(b_path))
    worst = 0
    worst_frame = -1
    count = 0
    try:
        while True:
            ok_a, fa = a.read()
            ok_b, fb = b.read()
            if not ok_a or not ok_b:
                if ok_a != ok_b:
                    raise RuntimeError("the two arms produced different frame counts")
                break
            if fa.shape != fb.shape:
                raise RuntimeError(f"frame shapes differ: {fa.shape} vs {fb.shape}")
            diff = int(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())
            if diff > worst:
                worst, worst_frame = diff, count
            count += 1
    finally:
        a.release()
        b.release()
    return worst, worst_frame, count


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--runs", type=int, default=3, help="warm runs per arm")
    parser.add_argument("--quality", default="PREVIEW", help="a video preset name")
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "algan_outputs" / "torch_compile_ab",
    )
    args = parser.parse_args()

    _register_test_fonts()

    import torch

    import algan
    from algan import SETTINGS, Scene
    from algan.scene_manager import SceneManager
    from algan.utils.torch_compile import (
        compiled_functions,
        torch_compile_enabled,
        torch_compile_support,
    )

    quality = getattr(algan, args.quality)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.set(
        output_root=str(out_dir),
        output_directory=".",
        cache_directory=str(out_dir / "cache"),
    )
    # Pin the frame-window split so the arms render the same batches (see
    # tests/fast/test_fast_render.py for why free memory would otherwise move
    # it between runs, and pixels with it).
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    os.chdir(args.scene.parent)

    supported, reason = torch_compile_support()
    device = SETTINGS.computing.render_device
    print(f"platform      : {platform.platform()} python {platform.python_version()}")
    print(f"torch         : {torch.__version__}")
    print(f"render device : {device}")
    print(f"compile ok?   : {supported}{'' if supported else ' -- ' + reason}")
    print(f"scene         : {args.scene}")
    print(f"quality       : {args.quality}", flush=True)

    def render(arm, tag):
        SETTINGS.computing.set(torch_compile=arm)
        SceneManager.reset()
        with Scene() as scene:
            _load_scene(args.scene)
            assert torch_compile_enabled() == arm
            started = time.perf_counter()
            scene.save_video(
                out_dir / f"{tag}.mp4",
                video_settings=quality,
                overwrite=True,
                animate_fade_out=True,
                codec="libx264rgb",
                ffmpeg_params=["-crf", "0", "-preset", "ultrafast"],
            )
            elapsed = time.perf_counter() - started
        print(
            f"  {tag:<12} {'compiled' if arm else 'eager':<9} {elapsed:7.2f}s",
            flush=True,
        )
        return elapsed

    results = {"eager": [], "compiled": []}
    print("cold (each arm's first render pays its one-off compile):")
    cold_eager = render(False, "cold_eager")
    cold_compiled = render(True, "cold_compiled")
    print("warm, alternating:")
    for i in range(args.runs):
        results["eager"].append(render(False, f"eager_{i}"))
        results["compiled"].append(render(True, f"compiled_{i}"))

    worst, worst_frame, frames = _compare_videos(
        out_dir / f"eager_{args.runs - 1}.mp4",
        out_dir / f"compiled_{args.runs - 1}.mp4",
    )
    eager = statistics.median(results["eager"])
    comp = statistics.median(results["compiled"])
    speedup = eager / comp if comp else float("nan")
    print()
    print(f"eager    warm: median {eager:.2f}s  min {min(results['eager']):.2f}s")
    print(f"compiled warm: median {comp:.2f}s  min {min(results['compiled']):.2f}s")
    print(f"speedup      : {speedup:.3f}x  (eager / compiled, medians)")
    print(f"cold         : eager {cold_eager:.2f}s, compiled {cold_compiled:.2f}s")
    verdict = "OK" if worst <= MAX_CHANNEL_DIFFERENCE else "EXCEEDS TOLERANCE"
    print(
        f"parity       : max channel difference {worst} over {frames} frames "
        f"(worst at frame {worst_frame}) -- {verdict}"
    )
    states = compiled_functions()
    compiled_count = sum(1 for _, state in states if state == "compiled")
    failed = [(name, state) for name, state in states if state.startswith("failed")]
    print(f"functions    : {compiled_count} compiled, {len(failed)} fell back to eager")
    for name, state in failed:
        print(f"  {name}: {state}")

    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "render_device": str(device),
                    "compile_supported": supported,
                    "scene": str(args.scene),
                    "quality": args.quality,
                    "cold": {"eager": cold_eager, "compiled": cold_compiled},
                    "warm": results,
                    "speedup": speedup,
                    "parity_max_channel_difference": worst,
                    "frames": frames,
                    "functions": states,
                },
                indent=2,
            )
        )
    return 0 if worst <= MAX_CHANNEL_DIFFERENCE else 1


if __name__ == "__main__":
    sys.exit(main())
