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
        [--runs 3] [--quality PREVIEW] [--json out.json] [--pn-controls]

``--pn-controls`` adds a third arm: the shipped compiled set *plus* the three
PN control-net builders that ``rendering/logical_pn.py`` deliberately leaves
eager (``logical_pn_control_points``,
``logical_pn_normal_control_points``, ``logical_pn_edge_control_points``),
compiled by patching their binding sites for the duration of that arm. It
exists to price the decision, not to reverse it: the arm is expected to differ
from the other two by more than the tolerance, because an ulp in the control
net moves a subdivision level. Point it at ``benchmarks/_pn_geometry_scene.py``
-- ``tests/fast/scene.py`` has no PN geometry at all, so the arm is a no-op
there.

The scene file only *records* an animation, exactly like ``tests/fast/scene.py``;
this harness owns the Scene, the settings and the comparison. It is rebuilt for
every run so no run inherits another's timeline. The two arms are compared
frame by frame on lossless video, against the same tolerance the render suites
use (``tests/conftest.py``, ``MAX_CHANNEL_DIFFERENCE``).
"""

from __future__ import annotations

import argparse
import contextlib
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


#: The three functions ``rendering/logical_pn.py`` keeps eager on purpose, and
#: every module that imported them by name (patching the defining module alone
#: would leave those bindings pointing at the eager originals).
_PN_CONTROL_FUNCTIONS = (
    "logical_pn_control_points",
    "logical_pn_normal_control_points",
    "logical_pn_edge_control_points",
)
_PN_CONTROL_MODULES = (
    "algan.rendering.logical_pn",
    "algan.rendering.raytracing.primitives",
    "algan.mobs.surfaces.surface",
)


def _build_pn_control_wrappers():
    """``{name: compiled wrapper}`` for the three eager-by-design builders.

    Built once and reused across arms, so the compiled arm pays Dynamo's build
    on its cold render and not again -- exactly as the shipped decorations do.
    """
    import importlib

    from algan.utils.torch_compile import compiled

    source = importlib.import_module("algan.rendering.logical_pn")
    return {name: compiled(getattr(source, name)) for name in _PN_CONTROL_FUNCTIONS}


@contextlib.contextmanager
def _pn_controls_compiled(wrappers):
    """Point every binding of the three builders at their compiled wrappers."""
    import importlib

    saved = []
    for module_name in _PN_CONTROL_MODULES:
        module = importlib.import_module(module_name)
        for name, wrapper in wrappers.items():
            if not hasattr(module, name):
                continue
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, wrapper)
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


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
        "--pn-controls",
        action="store_true",
        help="add a third arm that also compiles the PN control-net builders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "algan_outputs" / "torch_compile_ab",
    )
    args = parser.parse_args()

    _register_test_fonts()
    # Absolute before the chdir below, so a relative --scene, --output-dir or
    # --json given from the repository root still means what it said.
    args.scene = args.scene.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.json is not None:
        args.json = args.json.resolve()

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

    arms = ["eager", "compiled"]
    pn_wrappers = None
    if args.pn_controls:
        arms.append("compiled_pn")
        pn_wrappers = _build_pn_control_wrappers()

    def render(arm, tag):
        on = arm != "eager"
        SETTINGS.computing.set(torch_compile=on)
        SceneManager.reset()
        patch = (
            _pn_controls_compiled(pn_wrappers)
            if arm == "compiled_pn"
            else contextlib.nullcontext()
        )
        with patch, Scene() as scene:
            _load_scene(args.scene)
            assert torch_compile_enabled() == on
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
        print(f"  {tag:<14} {arm:<12} {elapsed:7.2f}s", flush=True)
        return elapsed

    results = {arm: [] for arm in arms}
    cold = {}
    print("cold (each arm's first render pays its one-off compile):")
    for arm in arms:
        cold[arm] = render(arm, f"cold_{arm}")
    print("warm, alternating:")
    for i in range(args.runs):
        for arm in arms:
            results[arm].append(render(arm, f"{arm}_{i}"))

    medians = {arm: statistics.median(results[arm]) for arm in arms}
    eager = medians["eager"]
    print()
    for arm in arms:
        print(
            f"{arm:<12} warm: median {medians[arm]:.2f}s  "
            f"min {min(results[arm]):.2f}s  "
            f"speedup {eager / medians[arm] if medians[arm] else float('nan'):.3f}x  "
            f"cold {cold[arm]:.2f}s"
        )
    speedup = eager / medians["compiled"] if medians["compiled"] else float("nan")

    parity = {}
    worst = 0
    for arm in arms[1:]:
        diff, worst_frame, frames = _compare_videos(
            out_dir / f"eager_{args.runs - 1}.mp4",
            out_dir / f"{arm}_{args.runs - 1}.mp4",
        )
        parity[arm] = diff
        verdict = "OK" if diff <= MAX_CHANNEL_DIFFERENCE else "EXCEEDS TOLERANCE"
        print(
            f"parity vs eager ({arm}): max channel difference {diff} over "
            f"{frames} frames (worst at frame {worst_frame}) -- {verdict}"
        )
        if arm == "compiled":
            worst = diff
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
                    "cold": cold,
                    "warm": results,
                    "medians": medians,
                    "speedup": speedup,
                    "parity_max_channel_difference": worst,
                    "parity": parity,
                    "functions": states,
                },
                indent=2,
            )
        )
    return 0 if worst <= MAX_CHANNEL_DIFFERENCE else 1


if __name__ == "__main__":
    sys.exit(main())
