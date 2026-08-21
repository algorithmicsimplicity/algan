"""Parity check for ``Scene.use_manim_defaults()``.

Renders one scene twice -- once by Manim, once by Algan through
:class:`~algan.mobs.manim_mob.ManimMob` -- and reports how far apart the two
videos are, per channel and per frame.

The two renders share a single list of Manim Mobjects (:func:`build_mobjects`),
so geometry, colour and placement cannot drift between them: what is being
measured is the *setting*, which is what ``use_manim_defaults()`` sets.

The motion is built **pose by pose**: frame *k* is a still render of the scene at
``t = k / frames``, in each engine, and the two sets of stills are stitched with
the same encoder settings. Playing one animation in each engine instead would
measure Manim's easing against Algan's rather than their cameras against each
other, and neither engine's animation system is what this helper touches.

Run it::

    <venv-python> benchmarks/_manim_defaults_parity_check.py            # 854x480, 12 frames
    <venv-python> benchmarks/_manim_defaults_parity_check.py --hd       # 1920x1080
    <venv-python> benchmarks/_manim_defaults_parity_check.py --frame    # single frame, fastest

Outputs land in ``algan_outputs/manim_parity/``: the two videos, a side-by-side
montage and an amplified difference image for whatever the numbers do not convey.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("algan_outputs/manim_parity")

# Manim's own default; both engines are pinned to it so a frame index means the
# same instant in each.
FPS = 30
DURATION_FRAMES = 12


def build_mobjects(manim, t=0.0):
    """Build the Mobjects under test at phase ``t``, in Manim's coordinate system.

    Both renderers consume this one list, so any difference in the output comes
    from the renderer or its settings rather than from the scene.

    The z coordinates are deliberately non-zero and the 3-D solids deliberately
    rotated: at ``z = 0`` Manim's perspective factor is exactly 1, so a flat
    scene would agree even with the field of view wrong. Sweeping ``t`` turns the
    solids through a full revolution and swings them through a range of depths,
    which is what actually exercises the perspective divide.

    Parameters
    ----------
    manim
        The imported ``manim`` module.
    t
        Phase of the motion, in ``[0, 1)``. Defaults to ``0.0``.

    Returns
    -------
    list
        The Manim Mobjects making up the scene at that phase.
    """
    m = manim
    DEG = m.DEGREES
    spin = 360.0 * t
    swing = math.sin(2 * math.pi * t)

    mobs = [
        # --- 2-D vector geometry, on the z = 0 plane -------------------------
        m.Square(side_length=1.6).shift(m.LEFT * 5.4 + m.UP * 2.6),
        m.Circle(radius=0.8).shift(m.LEFT * 3.2 + m.UP * 2.6),
        m.Triangle().scale(0.8).shift(m.LEFT * 1.1 + m.UP * 2.6),
        m.Star(outer_radius=0.85, color=m.YELLOW).shift(m.RIGHT * 1.1 + m.UP * 2.6),
        m.Annulus(inner_radius=0.35, outer_radius=0.8, color=m.TEAL).shift(
            m.RIGHT * 3.2 + m.UP * 2.6
        ),
        m.Polygon(
            m.ORIGIN, m.RIGHT * 1.5, m.RIGHT * 0.9 + m.UP * 1.4, color=m.PURPLE
        ).shift(m.RIGHT * 4.8 + m.UP * 1.9),
        # --- 2-D shapes with fills, strokes and text ------------------------
        m.Square(side_length=1.4, fill_color=m.GREEN, fill_opacity=0.9).shift(
            m.LEFT * 5.4 + m.UP * 0.4
        ),
        m.Circle(radius=0.7, fill_color=m.MAROON, fill_opacity=0.55).shift(
            m.LEFT * 3.2 + m.UP * 0.4
        ),
        m.Line(m.LEFT * 0.9, m.RIGHT * 0.9).shift(m.LEFT * 0.6 + m.UP * 0.7),
        m.Arrow(m.LEFT * 0.9, m.RIGHT * 0.9).shift(m.LEFT * 0.6 + m.UP * 0.0),
        m.Dot(color=m.ORANGE, radius=0.16).shift(m.RIGHT * 1.1 + m.UP * 0.4),
        m.MathTex(r"\int_0^\pi \sin x\,dx = 2")
        .scale(0.8)
        .shift(m.RIGHT * 3.6 + m.UP * 0.4),
        # --- 3-D solids, off the z = 0 plane and pre-rotated ------------------
        # These are what actually exercise the field of view, the perspective
        # divide and the z convention: each sits at a different depth.
        m.Sphere(radius=0.75)
        .rotate((70 + spin) * DEG, axis=m.RIGHT)
        .shift(m.LEFT * 5.0 + m.DOWN * 2.2 + m.OUT * (1.2 + 1.6 * swing)),
        m.Cube(side_length=1.3)
        .rotate((35 + spin) * DEG, axis=m.UP)
        .rotate(20 * DEG, axis=m.RIGHT)
        .shift(m.LEFT * 2.6 + m.DOWN * 2.2 + m.IN * (1.5 + 1.6 * swing)),
        m.Cylinder(radius=0.5, height=1.4)
        .rotate((55 + spin) * DEG, axis=m.RIGHT)
        .shift(m.DOWN * 2.2 + m.OUT * (2.4 - 1.6 * swing)),
        m.Cone(base_radius=0.6, height=1.4)
        .rotate((40 + spin) * DEG, axis=m.RIGHT)
        .shift(m.RIGHT * 2.4 + m.DOWN * 2.2 + m.IN * (2.5 - 1.6 * swing)),
        m.Torus(major_radius=0.7, minor_radius=0.22)
        .rotate((60 + spin) * DEG, axis=m.RIGHT)
        .shift(m.RIGHT * 5.0 + m.DOWN * 2.2 + m.OUT * (1.4 * swing)),
    ]
    return mobs


# --------------------------------------------------------------------------
# Rendering: one still per pose, in each engine
# --------------------------------------------------------------------------

_MANIM_DRIVER = """
import sys
sys.path.insert(0, {repo!r})
from manim import ThreeDScene
import manim as _m
from benchmarks._manim_defaults_parity_check import build_mobjects

PHASE = {phase!r}


class ParityScene(ThreeDScene):
    def construct(self):
        for mob in build_mobjects(_m, PHASE):
            self.add(mob)
"""


def render_manim_pose(width, height, phase, workdir, index):
    """Render one pose with Manim's own renderer; return the PNG path."""
    driver = Path(workdir) / f"pose_{index:04d}.py"
    driver.write_text(
        _MANIM_DRIVER.format(
            repo=str(Path(__file__).resolve().parent.parent), phase=float(phase)
        )
    )
    media = Path(workdir) / f"media_{index:04d}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "--resolution",
            f"{width},{height}",
            "--media_dir",
            str(media),
            "--disable_caching",
            "-s",
            "--format=png",
            str(driver),
            "ParityScene",
        ],
        check=True,
        cwd=workdir,
        stdout=subprocess.DEVNULL,
    )
    hits = sorted(media.glob("**/*.png"))
    if not hits:
        raise RuntimeError(f"manim produced no png for pose {index}")
    return hits[-1]


def render_algan_poses(width, height, phases, out_dir):
    """Render every pose through Algan with the Manim defaults; return the PNG paths."""
    import manim as _m

    from algan import Scene, VideoSettings
    from algan.mobs.manim_mob import ManimMob

    settings = VideoSettings(resolution=(width, height), frames_per_second=FPS)
    scene = Scene(video_settings=settings)
    paths = []
    for index, phase in enumerate(phases):
        # A fresh timeline per pose: this harness deliberately renders stills, so
        # nothing from the previous pose may survive into this one.
        scene.reset()
        scene.use_manim_defaults()
        for mob in build_mobjects(_m, phase):
            ManimMob(mob, scene=scene).spawn(animate=False)
        result = scene.save_frame(
            str((out_dir / f"algan_{index:04d}").resolve()), settings, overwrite=True
        )
        paths.append(Path(result.output_path))
    return paths


def stitch(frame_paths, out_path):
    """Encode a list of PNG frames into a video, identically for both engines."""
    listing = out_path.parent / f"{out_path.stem}_frames.txt"
    listing.write_text(
        "".join(
            f"file '{Path(p).resolve()}'\nduration {1 / FPS}\n" for p in frame_paths
        )
        + f"file '{Path(frame_paths[-1]).resolve()}'\n"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
    )
    listing.unlink()
    return out_path


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def read_frames(path, limit=None):
    """Read a video or image into a ``[frame, row, column, rgb]`` uint8 array."""
    import cv2

    path = str(path)
    if path.lower().endswith(".png"):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read {path}")
        return np.asarray([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)])

    capture = cv2.VideoCapture(path)
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"could not read any frame from {path}")
    return np.asarray(frames)


def compare(manim_frames, algan_frames, report_dir):
    """Report how far apart the two renders are, and write the diagnostic images."""
    import cv2

    a = np.concatenate([read_frames(p) for p in manim_frames])
    b = np.concatenate([read_frames(p) for p in algan_frames])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    if a.shape != b.shape:
        raise SystemExit(f"frame size mismatch: manim {a.shape} vs algan {b.shape}")

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_frame_max = diff.reshape(n, -1).max(axis=1)
    per_frame_mean = diff.reshape(n, -1).mean(axis=1)

    # A channel is "visibly" off if a viewer could plausibly see it. 2 is the
    # tolerance the render suites use for run-to-run noise; 16 is about where a
    # flat-colour difference stops being invisible on a black background.
    frac_over_2 = float((diff > 2).mean())
    frac_over_16 = float((diff > 16).mean())

    print(f"\nframes compared          : {n}")
    print(f"max channel difference   : {int(diff.max())}")
    print(f"mean channel difference  : {diff.mean():.3f}")
    print(f"pixels-channels > 2      : {frac_over_2 * 100:.3f}%")
    print(f"pixels-channels > 16     : {frac_over_16 * 100:.3f}%")
    print(f"per-frame max            : {per_frame_max.tolist()}")
    print(
        f"per-frame mean           : {[round(v, 3) for v in per_frame_mean.tolist()]}"
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    montage = np.concatenate([a[0], b[0]], axis=0)
    cv2.imwrite(
        str(report_dir / "montage_manim_over_algan.png"),
        cv2.cvtColor(montage, cv2.COLOR_RGB2BGR),
    )
    amplified = np.clip(diff[0].astype(np.int32) * 8, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(report_dir / "difference_x8.png"),
        cv2.cvtColor(amplified, cv2.COLOR_RGB2BGR),
    )
    print(f"\nwrote montage and 8x-amplified difference to {report_dir}/")
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hd", action="store_true", help="render at 1920x1080")
    parser.add_argument(
        "--frames", type=int, default=DURATION_FRAMES, help="how many poses to render"
    )
    args = parser.parse_args()

    width, height = (1920, 1080) if args.hd else (854, 480)
    report_dir = OUTPUT_DIR
    frame_dir = report_dir / "frames"
    for directory in (report_dir, frame_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phases = [k / args.frames for k in range(args.frames)]
    workdir = tempfile.mkdtemp(prefix="manim_parity_")
    try:
        print(f"rendering {len(phases)} poses with manim at {width}x{height} ...")
        manim_frames = []
        for index, phase in enumerate(phases):
            source = render_manim_pose(width, height, phase, workdir, index)
            kept = frame_dir / f"manim_{index:04d}.png"
            shutil.copy(source, kept)
            manim_frames.append(kept)
            print(f"  pose {index + 1}/{len(phases)}", end="\r", flush=True)

        print(f"\nrendering {len(phases)} poses with algan at {width}x{height} ...")
        algan_frames = render_algan_poses(width, height, phases, frame_dir)

        manim_video = stitch(manim_frames, report_dir / "manim.mp4")
        algan_video = stitch(algan_frames, report_dir / "algan.mp4")
        print(f"\nvideos: {manim_video}  {algan_video}")

        # Compare the source frames, not the encoded videos: h.264 is lossy, so
        # comparing the mp4s would measure the encoder as well as the renderers.
        compare(manim_frames, algan_frames, report_dir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("ALGAN_USE_DAEMON", "0")
    main()
