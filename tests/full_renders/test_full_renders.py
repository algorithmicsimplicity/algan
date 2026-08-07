"""Pixel-comparison harness for the full-render scenes.

Each file in ``scenes/`` authors one dense Scene covering a whole subsystem.
This module renders it at ``PREVIEW`` and compares every frame against the
checked-in baseline in ``expected_outputs_<device>/``.

Re-baselining
-------------
A rendering change that legitimately alters output is re-baselined by rendering
with the baselines writable and *looking at the result* before committing it::

    ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/Scripts/python.exe -m pytest \
        tests/full_renders -q

On Windows, render work must run one process at a time: a killed run orphans
child processes that keep the output mp4s locked.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from algan import PREVIEW, SETTINGS, Scene
from algan.scene_manager import SceneManager

HERE = Path(__file__).resolve().parent
SCENES_DIR = HERE / "scenes"
OUTPUT_DIR = HERE / "algan_outputs"
CACHE_DIR = HERE / "algan_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_DIR = HERE / f"expected_outputs_{DEVICE}"
UPDATE_BASELINES = os.getenv("ALGAN_UPDATE_FULL_RENDER_BASELINES") == "1"

# Small per-pixel drift is expected and tolerated: torch CPU rate-function
# evaluation rounds differently depending on the materialization window, so
# byte-identity across re-windowed state is unattainable.
MAX_CHANNEL_DIFFERENCE = 2

SCENE_FILES = sorted(
    path for path in SCENES_DIR.glob("*.py") if not path.name.startswith("_")
)


@pytest.fixture
def render_environment(monkeypatch):
    """Isolate one scene render from the process and from its neighbours.

    Scenes are allowed to write to ``SETTINGS`` (``materials_and_lighting``
    turns shadows on), so the whole settings root is snapshotted and restored.
    The working directory becomes ``tests/full_renders`` so a scene can name its
    assets relative to itself -- ``resolve_asset_path`` tries the working
    directory first.
    """
    snapshot = SETTINGS.snapshot()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(HERE)
    SETTINGS.paths.set(
        output_root=str(HERE),
        output_directory=OUTPUT_DIR.name,
        cache_directory=str(CACHE_DIR),
    )
    SceneManager.reset()
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


def _load_scene(scene_path: Path) -> None:
    """Execute a scene file, which records (but never renders) its animation."""
    module_name = f"_algan_full_render_{scene_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, scene_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load full-render scene {scene_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Scene modules are re-executed per run; never leave them importable.
        sys.modules.pop(module_name, None)


def _compare_videos(actual_path: Path, expected_path: Path, diff_path: Path) -> None:
    actual = cv2.VideoCapture(str(actual_path))
    expected = cv2.VideoCapture(str(expected_path))
    expected_fps = expected.get(cv2.CAP_PROP_FPS) or PREVIEW.frames_per_second
    writer = None
    frame_count = 0
    max_difference = 0
    worst_frame = -1

    try:
        while True:
            actual_ok, actual_frame = actual.read()
            expected_ok, expected_frame = expected.read()
            if not actual_ok or not expected_ok:
                assert actual_ok == expected_ok, (
                    f"{actual_path.name} has a different frame count from its "
                    f"baseline (diverged at frame {frame_count})"
                )
                break

            assert actual_frame.shape == expected_frame.shape, (
                f"{actual_path.name} rendered at {actual_frame.shape}, expected "
                f"{expected_frame.shape}"
            )
            difference = np.abs(
                actual_frame.astype(np.int16) - expected_frame.astype(np.int16)
            ).astype(np.uint8)
            frame_difference = int(difference.max())
            if frame_difference > max_difference:
                max_difference = frame_difference
                worst_frame = frame_count
            frame_count += 1

            if frame_difference > MAX_CHANNEL_DIFFERENCE:
                if writer is None:
                    diff_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = difference.shape[:2]
                    writer = cv2.VideoWriter(
                        str(diff_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        expected_fps,
                        (width, height),
                    )
                writer.write(difference)
    finally:
        actual.release()
        expected.release()
        if writer is not None:
            writer.release()

    assert frame_count > 0, f"{actual_path.name} did not contain any frames"
    assert max_difference <= MAX_CHANNEL_DIFFERENCE, (
        f"{actual_path.name} differs from its baseline by up to "
        f"{max_difference} channel values (worst at frame {worst_frame}); "
        f"see {diff_path}"
    )


def test_there_is_at_least_one_full_render_scene():
    """A silently empty scene directory would make the whole suite vacuous."""
    assert SCENE_FILES, f"no full-render scenes found in {SCENES_DIR}"


@pytest.mark.slow
@pytest.mark.parametrize("scene_path", SCENE_FILES, ids=lambda path: path.stem)
def test_full_render_scene(scene_path: Path, render_environment):
    output_path = OUTPUT_DIR / f"{scene_path.stem}.mp4"
    output_path.unlink(missing_ok=True)

    with Scene() as scene:
        _load_scene(scene_path)
        result = scene.save_video(
            output_path,
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
        )

    assert result.rendered
    assert result.output_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    expected_path = EXPECTED_DIR / output_path.name
    if UPDATE_BASELINES:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, expected_path)
        pytest.skip(f"re-baselined {output_path.name}")
    if not EXPECTED_DIR.exists():
        pytest.skip(f"no {DEVICE} full-render baselines are available")

    assert expected_path.exists(), (
        f"Missing baseline for {scene_path.name}. Re-run with "
        "ALGAN_UPDATE_FULL_RENDER_BASELINES=1 after reviewing the render."
    )
    _compare_videos(
        output_path,
        expected_path,
        HERE / "output_errors" / output_path.name,
    )
