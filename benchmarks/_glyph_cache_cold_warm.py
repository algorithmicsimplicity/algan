"""Does the glyph geometry cache change the render? -- the trap ssB.2 fell into.

``CLAUDE.md`` records this for a cloud container: the FIRST render on a machine
populates the Manim Tex/SVG geometry cache, and its glyph antialiasing differs
from every run after it -- 18 channel values over 100 frames of
``text_and_media``, against a tolerance of 2. The same hazard is restated in
``DESIGN_mesh_identity_open.md`` ssA for a local kernel-constant A/B, because
``clear_cache(taichi_kernels=True)`` deletes the WHOLE ``~/.algan/cache``,
glyph geometry included, so the render right after a cache wipe differs from
every later one *for a reason that has nothing to do with the change under
test* -- and it lands in the diff looking exactly like the change.

Nothing measured it for a scene other than ``text_and_media``, so nobody could
say whether a given A/B's residue was this or the change. This measures it, for
any of the six full-render scenes, without deleting anything: the cache
DIRECTORY is redirected to a fresh path, so the cold arm is cold by
construction and ``~/.algan/cache`` is never touched.

    cold   render into an empty cache directory  (parses glyph geometry)
    warm   render again into the same directory  (replays it)

A non-zero diff means any A/B whose arms straddle a cache wipe carries this on
top of whatever it was measuring.

Usage:
    <venv-python> benchmarks/_glyph_cache_cold_warm.py shapes_and_timeline
    <venv-python> benchmarks/_glyph_cache_cold_warm.py text_and_media --res md
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FULL_RENDERS = REPO / "tests" / "full_renders"

#: What the suite pins, so the frame windows fall where the baselines' did.
PINNED_BYTES = 1536 * 1024 * 1024


def _render(scene_name, out_path, cache_dir, res):
    """Render one full-render scene with the cache pointed somewhere specific."""
    import torch  # noqa: F401,PLC0415

    from algan import HD, LD, MD, PREVIEW, SETTINGS, Scene  # noqa: PLC0415
    from algan.scene_manager import SceneManager  # noqa: PLC0415

    quality = {"ld": LD, "md": MD, "hd": HD, "preview": PREVIEW}[res]
    conftest = REPO / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_algan_glyph_conf", conftest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_register_bundled_fonts", "register_test_fonts"):
        fn = getattr(module, name, None)
        if callable(fn):
            fn()
            break

    os.chdir(FULL_RENDERS)
    SETTINGS.paths.set(
        output_root=str(FULL_RENDERS),
        output_directory=str(Path(out_path).parent),
        cache_directory=str(cache_dir),
    )
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    SceneManager.reset()
    path = FULL_RENDERS / "scenes" / f"{scene_name}.py"
    with Scene() as scene:
        spec = importlib.util.spec_from_file_location(f"_glyph_{scene_name}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(f"_glyph_{scene_name}", None)
        scene.save_video(
            str(out_path),
            video_settings=quality,
            overwrite=True,
            animate_fade_out=True,
        )


def _diff(a, b):
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    worst = worst_moved = frames = total_moved = 0
    worst_frame = -1
    while True:
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not ok_a or not ok_b:
            break
        d = abs(fa.astype(np.int16) - fb.astype(np.int16))
        m = int((d.max(axis=2) > 2).sum())
        if int(d.max()) > worst:
            worst, worst_frame = int(d.max()), frames
        worst_moved = max(worst_moved, m)
        total_moved += m
        frames += 1
    ca.release()
    cb.release()
    return worst, worst_frame, worst_moved, total_moved, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene")
    ap.add_argument("--res", default="preview", choices=("ld", "md", "hd", "preview"))
    ap.add_argument(
        "--keep", default=None, help="write the two videos here instead of a temp dir"
    )
    ap.add_argument(
        "--seed-tex",
        action="store_true",
        help="copy the existing cache's manim/ tree (Tex -> SVG output) into "
        "the fresh directory, leaving only Algan's own PARSED geometry cache "
        "empty. That is the arm the ss6.x note is about -- glyph geometry "
        "PARSED vs REPLAYED -- and it does not need a working LaTeX, which a "
        "genuinely cold arm does",
    )
    ap.add_argument("--render", nargs=3, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.render:
        _render(args.scene, args.render[0], args.render[1], args.render[2])
        return 0

    work = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="algan_glyph_"))
    )
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    if cache.exists():
        # Only ever a directory this script itself made under `work`.
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    if args.seed_tex:
        from algan.settings import SETTINGS  # noqa: PLC0415

        src = Path(SETTINGS.paths.cache_directory) / "manim"
        if src.exists():
            shutil.copytree(src, cache / "manim")
            print(f"seeded Tex output from {src}")
        else:
            print(f"nothing to seed: {src} does not exist")
    outs = []
    # Each arm is its own interpreter: a warm arm sharing a process with the
    # cold one could replay from an in-memory memo and never touch the disk
    # cache, which would measure nothing.
    for arm in ("cold", "warm"):
        out = work / f"{args.scene}_{arm}.mp4"
        print(
            f"-- {arm} render (cache {'empty' if arm == 'cold' else 'populated'})",
            flush=True,
        )
        r = subprocess.run(
            [
                sys.executable,
                __file__,
                args.scene,
                "--res",
                args.res,
                "--render",
                str(out),
                str(cache),
                args.res,
            ],
            cwd=str(REPO),
        )
        if r.returncode != 0:
            print(f"   {arm} arm FAILED (exit {r.returncode})")
            return 1
        outs.append(out)

    worst, wf, worst_moved, total_moved, frames = _diff(*outs)
    print(
        f"\n{args.scene} @{args.res}: cold vs warm -> max |d| {worst} "
        f"(frame {wf}), worst frame {worst_moved} px over tol, "
        f"{total_moved} px total over {frames} frames"
    )
    print(f"videos kept in {work}")
    if worst:
        print(
            "\nThe glyph cache alone moves this scene. Any A/B whose arms "
            "straddle a cache wipe carries this on top of what it measured."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
