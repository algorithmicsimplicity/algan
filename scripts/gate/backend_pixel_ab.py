"""Render the full-render scenes under each kernel compiler and diff the pixels.

The question this answers is the one `taichi_patches/PLAN.md` §6 says decides
the base: **how far do the pixels move when the same scene is compiled by
Quadrants (LLVM 22) instead of Taichi 1.7.4 (LLVM 15)?** Not against a
baseline -- against the *other backend on the same box*, which is the only
comparison that isolates the compiler.

Why not just run `pytest tests/full_renders`: those baselines are
machine-specific. `tests/full_renders/test_full_renders.py:76-92` measured five
of the six scenes differing by 29-204 channel values between two machines on
one Taichi, because `fast_math` flips borderline tessellation levels and which
ones are borderline depends on the hardware. So on a T4 or a Mac runner the
suite fails on master and tells you nothing about a backend swap. Two arms on
one box, compared to each other, does.

    # one arm, in this process's backend
    ALGAN_TAICHI_BACKEND=taichi python scripts/gate/backend_pixel_ab.py \
        --render --out /tmp/ab/taichi

    # both arms, sequentially, then the diff  (the usual entry point)
    python scripts/gate/backend_pixel_ab.py --both --workdir /tmp/ab

    # diff two directories rendered earlier
    python scripts/gate/backend_pixel_ab.py --compare /tmp/ab/taichi /tmp/ab/quadrants

`--both` runs the arms **sequentially, in separate processes**, and both are
load-bearing. The backend is bound at first use and cannot be re-selected in a
live process (`algan/taichi_compat.py`), and free VRAM at render time sets the
frame-window split -- two concurrent arms would change each other's pixels
(`agent_guidance/gpu_harnesses.md`, "Never take a determinism or pixel reading
while another process is using the GPU").

Everything the render suite pins to keep a render reproducible is pinned here
the same way and for the same reasons: PREVIEW, `available_memory_override` at
1.5 GiB, lossless `libx264rgb -crf 0`, `animate_fade_out=True`, and the working
directory at `tests/full_renders` so scene assets resolve. A difference in any
of those would show up as a pixel delta and be read as a compiler difference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Both are set before `import algan`: the daemon keeps adaptive renderer state
# across runs, so an arm served by it would be rendered against whatever ran
# before it -- and the two arms must not share a process at all, since the
# kernel compiler is bound once per process.
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = REPO_ROOT / "tests" / "full_renders" / "scenes"
# The suite's own figure, and it must stay the suite's figure: it replaces the
# free-memory measurement that would otherwise split frames differently on
# every box (`test_full_renders.py:63-83`).
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024
# `tests/conftest.py`'s tolerance, so a verdict here means what it means there.
MAX_CHANNEL_DIFFERENCE = 2


def _scene_paths(names: list[str] | None) -> list[Path]:
    available = sorted(p for p in SCENES_DIR.glob("*.py") if not p.name.startswith("_"))
    if not names:
        return available
    by_stem = {path.stem: path for path in available}
    missing = [name for name in names if name not in by_stem]
    if missing:
        raise SystemExit(
            f"unknown scene(s): {', '.join(missing)}; "
            f"available: {', '.join(sorted(by_stem))}"
        )
    return [by_stem[name] for name in names]


def render_arm(out_dir: Path, names: list[str] | None) -> int:
    """Render every scene into ``out_dir`` with this process's backend."""
    import importlib.util

    import algan
    from algan import PREVIEW, SETTINGS, Scene
    from algan.scene_manager import SceneManager
    from algan.settings import _startup
    from algan.taichi_compat import describe_backend

    out_dir.mkdir(parents=True, exist_ok=True)
    device = _startup.render_device().type
    print(f"ARM backend={describe_backend()} device={device} algan={algan.__version__}")

    timings: dict[str, float] = {}
    for scene_path in _scene_paths(names):
        snapshot = SETTINGS.snapshot()
        # Scene assets are named relative to `tests/full_renders`, which is
        # where the suite chdirs before loading one.
        os.chdir(SCENES_DIR.parent)
        SETTINGS.paths.set(
            output_root=str(out_dir),
            output_directory=".",
            cache_directory=str(REPO_ROOT / "tests" / "full_renders" / "algan_cache"),
        )
        SETTINGS.computing.set(available_memory_override=AVAILABLE_MEMORY_OVERRIDE)
        SceneManager.reset()
        started = time.perf_counter()
        try:
            module_name = f"_algan_gate_{scene_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, scene_path)
            module = importlib.util.module_from_spec(spec)
            with Scene() as scene:
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.modules.pop(module_name, None)
                scene.save_video(
                    out_dir / f"{scene_path.stem}.mp4",
                    video_settings=PREVIEW,
                    overwrite=True,
                    animate_fade_out=True,
                    codec="libx264rgb",
                    ffmpeg_params=["-crf", "0", "-preset", "fast"],
                )
        finally:
            elapsed = time.perf_counter() - started
            timings[scene_path.stem] = elapsed
            SETTINGS.restore(snapshot)
            SceneManager.reset()
        print(f"  rendered {scene_path.stem} in {elapsed:.1f}s", flush=True)

    (out_dir / "arm.json").write_text(
        json.dumps(
            {"backend": describe_backend(), "device": device, "seconds": timings},
            indent=2,
        )
    )
    print(f"ARM-TOTAL {sum(timings.values()):.1f}s over {len(timings)} scenes")
    return 0


def compare(dir_a: Path, dir_b: Path) -> int:
    """Frame-by-frame diff of two arms, with `tests/conftest.py`'s tolerance."""
    import cv2
    import numpy as np

    videos = sorted(p.name for p in dir_a.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos in {dir_a}")

    print(f"{'scene':<32} {'frames':>6} {'max':>4} {'>tol px':>9} {'of':>11} worst")
    worst_overall = 0
    failures = []
    for name in videos:
        path_b = dir_b / name
        if not path_b.exists():
            print(f"{name:<32} MISSING in {dir_b}")
            failures.append(name)
            continue
        cap_a, cap_b = (
            cv2.VideoCapture(str(dir_a / name)),
            cv2.VideoCapture(str(path_b)),
        )
        frames = max_difference = over_tolerance = total_pixels = worst_frame = 0
        # Brightness per arm, because "how far apart" does not distinguish the
        # two failures that matter here. Float drift moves a few pixels a
        # little; a black frame -- the exact failure the MPS zero-copy patch
        # exists to prevent, and one a smoke test cannot see because the render
        # still completes -- moves nearly all of them the whole way. A mean
        # near zero on one arm and not the other says which, without anyone
        # downloading a video.
        sum_a = sum_b = 0.0
        try:
            while True:
                ok_a, frame_a = cap_a.read()
                ok_b, frame_b = cap_b.read()
                if not ok_a or not ok_b:
                    if ok_a != ok_b:
                        print(f"{name:<32} FRAME COUNT DIVERGED at {frames}")
                        failures.append(name)
                    break
                difference = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
                frame_max = int(difference.max())
                if frame_max > max_difference:
                    max_difference, worst_frame = frame_max, frames
                over_tolerance += int(
                    (difference > MAX_CHANNEL_DIFFERENCE).any(axis=2).sum()
                )
                total_pixels += difference.shape[0] * difference.shape[1]
                sum_a += float(frame_a.mean())
                sum_b += float(frame_b.mean())
                frames += 1
        finally:
            cap_a.release()
            cap_b.release()
        worst_overall = max(worst_overall, max_difference)
        if max_difference > MAX_CHANNEL_DIFFERENCE:
            failures.append(name)
        mean_a = sum_a / frames if frames else 0.0
        mean_b = sum_b / frames if frames else 0.0
        print(
            f"{name:<32} {frames:>6} {max_difference:>4} {over_tolerance:>9} "
            f"{total_pixels:>11} frame {worst_frame}"
        )
        print(
            f"{'':<32} mean brightness: A={mean_a:6.2f}  B={mean_b:6.2f}"
            + ("   <-- A is blank" if mean_a < 1.0 <= mean_b else "")
            + ("   <-- B is blank" if mean_b < 1.0 <= mean_a else "")
        )

    verdict = (
        "IDENTICAL"
        if worst_overall == 0
        else ("WITHIN-TOLERANCE" if not failures else "EXCEEDS-TOLERANCE")
    )
    print(
        f"GATE-RESULT: {verdict} max_channel_delta={worst_overall} "
        f"tolerance={MAX_CHANNEL_DIFFERENCE} scenes_over={len(set(failures))}"
    )
    return 0 if not failures else 1


def run_both(workdir: Path, names: list[str] | None, backends: list[str]) -> int:
    """Render each backend in its own process, sequentially, then compare."""
    for backend in backends:
        out_dir = workdir / backend
        print(f"\n=== arm: {backend} -> {out_dir} ===", flush=True)
        environment = dict(os.environ, ALGAN_TAICHI_BACKEND=backend)
        # The launch-path and frontend patches version-gate themselves to
        # Taichi 1.7 and no-op elsewhere; turning them off on both arms keeps
        # the comparison about codegen rather than about which arm got patched.
        environment.setdefault("ALGAN_TAICHI_WARMSTART", "0")
        environment.setdefault("ALGAN_TAICHI_FAST_LAUNCH", "0")
        environment.setdefault("QD_KERNEL_COVERAGE", "0")
        command = [sys.executable, __file__, "--render", "--out", str(out_dir)]
        if names:
            command += ["--scenes", *names]
        started = time.perf_counter()
        result = subprocess.run(command, env=environment, cwd=str(REPO_ROOT))
        print(
            f"=== arm {backend} exited {result.returncode} "
            f"in {time.perf_counter() - started:.1f}s ===",
            flush=True,
        )
        if result.returncode != 0:
            print(f"GATE-RESULT: ARM-FAILED backend={backend}")
            return result.returncode
    return compare(workdir / backends[0], workdir / backends[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true", help="render one arm")
    mode.add_argument("--both", action="store_true", help="both arms, then compare")
    mode.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    parser.add_argument("--out", type=Path, help="output directory for --render")
    parser.add_argument("--workdir", type=Path, help="parent directory for --both")
    parser.add_argument("--scenes", nargs="*", help="scene stems (default: all six)")
    parser.add_argument(
        "--backends",
        nargs=2,
        default=["taichi", "quadrants"],
        help="the two ALGAN_TAICHI_BACKEND values to compare",
    )
    args = parser.parse_args(argv)

    if args.render:
        if args.out is None:
            parser.error("--render needs --out")
        return render_arm(args.out.resolve(), args.scenes)
    if args.compare:
        return compare(Path(args.compare[0]).resolve(), Path(args.compare[1]).resolve())
    if args.workdir is None:
        parser.error("--both needs --workdir")
    return run_both(args.workdir.resolve(), args.scenes, args.backends)


if __name__ == "__main__":
    raise SystemExit(main())
