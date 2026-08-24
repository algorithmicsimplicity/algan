"""Render ``tests/full_renders/scenes/complex_hierarchy_become.py`` and dump frames.

The full-render suite compares this scene to a baseline; it says nothing about
whether the picture is *right*. This renders the same scene through the same
settings the suite uses and writes every frame as a PNG so the morph can be read
frame by frame.

    <venv-python> benchmarks/_hierarchy_become_probe.py [--out DIR] [--scene PATH]

Frames land in ``DIR/frame_XXXX.png`` (default: a ``_hierarchy_become_out``
directory under ``benchmarks/``), and the video beside them.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The scenes name their fonts; tests/conftest.py registers the vendored faces.
sys.path.insert(0, str(REPO / "tests"))

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024


def _register_fonts() -> None:
    import manimpango

    font_dir = REPO / "tests" / "assets" / "fonts"
    for face in sorted(font_dir.glob("*.ttf")):
        manimpango.register_font(str(face))


def _load_scene(scene_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("_probe_scene", scene_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_probe_scene", None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default=str(
            REPO / "tests" / "full_renders" / "scenes" / "complex_hierarchy_become.py"
        ),
    )
    parser.add_argument(
        "--out", default=str(REPO / "benchmarks" / "_hierarchy_become_out")
    )
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args()

    scene_path = Path(args.scene).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _register_fonts()

    here = REPO / "tests" / "full_renders"
    os.chdir(here)
    SETTINGS.paths.set(
        output_root=str(out_dir),
        output_directory=".",
        cache_directory=str(here / "algan_cache"),
    )
    SETTINGS.computing.set(available_memory_override=AVAILABLE_MEMORY_OVERRIDE)
    SceneManager.reset()

    video_path = out_dir / f"{scene_path.stem}.mp4"
    with Scene() as scene:
        _load_scene(scene_path)
        scene.save_video(
            video_path,
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )

    print(f"video: {video_path}")
    if not args.no_frames:
        frame_dir = out_dir / f"{scene_path.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for stale in frame_dir.glob("frame_*.png"):
            stale.unlink()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                str(frame_dir / "frame_%04d.png"),
            ],
            check=True,
        )
        print(f"frames: {frame_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
