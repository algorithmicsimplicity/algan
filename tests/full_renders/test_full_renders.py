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

import pytest
import torch

from algan import PREVIEW, SETTINGS, Scene
from algan.scene_manager import SceneManager

HERE = Path(__file__).resolve().parent
SCENES_DIR = HERE / "scenes"
OUTPUT_DIR = HERE / "algan_outputs"
CACHE_DIR = HERE / "algan_cache"
ERRORS_DIR = HERE / "output_errors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_DIR = HERE / f"expected_outputs_{DEVICE}"
UPDATE_BASELINES = os.getenv("ALGAN_UPDATE_FULL_RENDER_BASELINES") == "1"

# Frames are compared by the ``assert_video_matches_baseline`` fixture in
# ``tests/conftest.py``, which both render suites share so they cannot drift
# apart on tolerance.

# Free VRAM is what the render loop sizes its frame windows from, and it is not
# reproducible: it shrinks as the Torch and Taichi allocators warm up, so the
# same scene split differently depending on how many scenes had rendered before
# it in the process. That is not a tolerable drift -- a different split carries
# a different set of not-yet-spawned actors into the batch and pads the merged
# arrays to a different width, which reorders them and the STBVH, and
# silhouettes moved by up to 54 channel values between two splits of this very
# suite. Pinning the measurement makes each scene render the same way every
# time. The figure has to be affordable on the device (it replaces the
# measurement, it does not cap it); 1.5 GiB leaves a 600 MB arena, which every
# scene here renders inside.
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024

# These baselines are per *machine*, not merely per device, so this suite gates
# locally and on whichever machine rendered its baselines -- not across
# machines, and therefore not in CI.
#
# Measured, rather than assumed: a GitHub Actions ubuntu-latest runner rendered
# these scenes against baselines produced on another CPU and five of the six
# missed, by 29 (text_and_media), 44 (complex_hierarchy_become), 50
# (manim_compat_and_plots), 53 (solids_and_camera) and 204
# (materials_and_lighting) channel values, against a tolerance of 2. The two
# that matched -- shapes_and_timeline here, and tests/fast -- are the ones built
# from 2-D circuits and flat triangle meshes. Everything that moved carries PN
# surfaces, shadows, refraction or glTF, which is what ``pn_criterion_kernel``
# running under Taichi's ``fast_math`` would predict: it flips borderline
# tessellation levels, and which ones are borderline depends on the CPU.
#
# CI therefore runs tests/unit_tests and tests/fast, both of which are portable.
# Raising the tolerance to cover this would have to reach ~204 and would swallow
# the regressions the suite exists to catch, and re-baselining on a runner would
# only move the failure onto the developer's machine.
#
# Set ALGAN_RUN_FULL_RENDERS=1 to run it anyway -- on a machine whose baselines
# these are, or when deliberately re-measuring the cross-machine spread.
IN_CI = os.getenv("CI") == "true"
FORCE_FULL_RENDERS = os.getenv("ALGAN_RUN_FULL_RENDERS") == "1"
SKIP_IN_CI_REASON = (
    "full-render baselines are machine-specific (5/6 scenes differ by 29-204 "
    "channel values across machines); CI covers tests/unit_tests and tests/fast. "
    "Set ALGAN_RUN_FULL_RENDERS=1 to override."
)

SCENE_FILES = sorted(
    path for path in SCENES_DIR.glob("*.py") if not path.name.startswith("_")
)


@pytest.fixture(scope="module", autouse=True)
def _clear_output_errors():
    """Empty ``output_errors/`` once, before the first scene of a run.

    Diff videos are named after their scene, so a scene that passes this run
    leaves last run's diff sitting beside this run's. Clearing the directory up
    front makes its contents mean exactly "what failed in the most recent run".

    Module-scoped so the scenes of one run do not wipe each other's diffs. On
    Windows a diff video left open in a player cannot be deleted; ``rmtree``
    skips it rather than failing the suite over it, and the comparison fixture
    recreates the directory when it next writes.
    """
    shutil.rmtree(ERRORS_DIR, ignore_errors=True)


@pytest.fixture
def render_environment(monkeypatch):
    """Isolate one scene render from the process and from its neighbours.

    Scenes are allowed to write to ``SETTINGS`` (``materials_and_lighting``
    turns shadows on), so the whole settings root is snapshotted and restored.
    The working directory becomes ``tests/full_renders`` so a scene can name its
    assets relative to itself -- ``resolve_asset_path`` tries the working
    directory first.

    ``available_memory_override`` pins the frame-window split; see
    ``AVAILABLE_MEMORY_OVERRIDE``.
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
    SETTINGS.computing.set(available_memory_override=AVAILABLE_MEMORY_OVERRIDE)
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


def test_there_is_at_least_one_full_render_scene():
    """A silently empty scene directory would make the whole suite vacuous."""
    assert SCENE_FILES, f"no full-render scenes found in {SCENES_DIR}"


@pytest.mark.skipif(IN_CI and not FORCE_FULL_RENDERS, reason=SKIP_IN_CI_REASON)
@pytest.mark.parametrize("scene_path", SCENE_FILES, ids=lambda path: path.stem)
def test_full_render_scene(
    scene_path: Path, render_environment, assert_video_matches_baseline
):
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
    assert_video_matches_baseline(
        output_path,
        expected_path,
        ERRORS_DIR / output_path.name,
        fallback_fps=PREVIEW.frames_per_second,
    )
