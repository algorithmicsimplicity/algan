from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from algan import PREVIEW, SETTINGS, Scene

HERE = Path(__file__).resolve().parent
SCENES_DIR = HERE / "scenes"
OUTPUT_DIR = HERE / "algan_outputs"
CACHE_DIR = HERE / "algan_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_DIR = HERE / f"expected_outputs_{DEVICE}"
UPDATE_BASELINES = os.getenv("ALGAN_UPDATE_FULL_RENDER_BASELINES") == "1"
SCENE_FILES = sorted(
    path
    for path in SCENES_DIR.glob("*.py")
    if not path.name.startswith("_")
)


@pytest.fixture
def render_environment():
    previous_base = SETTINGS.paths.output_root
    previous_output = SETTINGS.paths.output_directory
    previous_cache = SETTINGS.paths.cache_directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.output_root = str(HERE)
    SETTINGS.paths.output_directory = OUTPUT_DIR.name
    SETTINGS.paths.cache_directory = str(CACHE_DIR)
    try:
        yield
    finally:
        SETTINGS.paths.output_root = previous_base
        SETTINGS.paths.output_directory = previous_output
        SETTINGS.paths.cache_directory = previous_cache


def _load_scene(scene_path: Path) -> None:
    module_name = f"_algan_full_render_{scene_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, scene_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load full-render scene {scene_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _compare_videos(actual_path: Path, expected_path: Path, diff_path: Path) -> None:
    actual = cv2.VideoCapture(str(actual_path))
    expected = cv2.VideoCapture(str(expected_path))
    expected_fps = expected.get(cv2.CAP_PROP_FPS) or PREVIEW.frames_per_second
    writer = None
    frame_count = 0
    max_difference = 0

    try:
        while True:
            actual_ok, actual_frame = actual.read()
            expected_ok, expected_frame = expected.read()
            if not actual_ok or not expected_ok:
                assert actual_ok == expected_ok, (
                    f"{actual_path.name} has a different frame count from its baseline"
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
            max_difference = max(max_difference, frame_difference)
            frame_count += 1

            if frame_difference > 2:
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
    assert max_difference <= 2, (
        f"{actual_path.name} differs from its baseline by up to "
        f"{max_difference} channel values; see {diff_path}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("scene_path", SCENE_FILES, ids=lambda path: path.stem)
def test_full_render_scene(scene_path: Path, render_environment):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    elif not EXPECTED_DIR.exists():
        pytest.skip(f"No {DEVICE} full-render baselines are available")

    assert expected_path.exists(), (
        f"Missing baseline for {scene_path.name}. Re-run with "
        "ALGAN_UPDATE_FULL_RENDER_BASELINES=1 after reviewing the render."
    )
    _compare_videos(
        output_path,
        expected_path,
        HERE / "output_errors" / output_path.name,
    )
