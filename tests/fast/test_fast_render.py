"""The fast suite's one render, compared pixel-wise against its baseline.

Everything else in the fast suite is behavioural: it asserts on tensors the
engine computed, never on what the renderer drew.  That leaves the largest
subsystem in the project — the tracer, the rasteriser, the shaders and the
video writer — visible only through the ten-minute ``tests/full_renders``
suite, which is too slow for a development loop.

So the fast suite renders exactly one scene (``scene.py``, which explains why
it is one) and compares every frame.  It costs about half the fast suite's
budget, almost all of it Taichi specialising the render kernel on this scene's
geometry, and it is the only thing standing between a development loop and a
renderer regression that no unit test can see.

Re-baselining
-------------
A change that legitimately alters output is re-baselined by rendering with the
baseline writable and *looking at the result* before committing it::

    ALGAN_UPDATE_FAST_BASELINE=1 .venv/Scripts/python.exe -m pytest tests/fast -q

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

from algan import PREVIEW, SETTINGS, Scene
from algan.scene_manager import SceneManager

HERE = Path(__file__).resolve().parent
SCENE_FILE = HERE / "scene.py"
OUTPUT_DIR = HERE / "algan_outputs"
CACHE_DIR = HERE / "algan_cache"
# The device the render will actually run on, which is not the same question as
# ``torch.cuda.is_available()``: a CUDA machine with ``ALGAN_RENDER_DEVICE=cpu``
# set renders on the CPU and belongs against the CPU baseline, and an Apple
# Silicon Mac renders on MPS while reporting no CUDA at all. Read at import,
# before any test can move it.
DEVICE = SETTINGS.computing.render_device.type
# macOS is keyed apart from the other platforms on the same device, and the
# reason is measured rather than assumed: the x86-64 CPU baseline was copied
# into ``expected_outputs_macos_cpu/`` and rendered against on an Apple Silicon
# CI runner, and it missed by up to 45 channel values (worst at frame 36)
# against a tolerance of 2. So this scene does *not* survive the change of
# instruction set, even though it is the one that matched exactly across two
# x86-64 machines -- fp32 arithmetic through a path tracer does not agree
# across two libm implementations that closely.
#
# Nothing is committed under that name, so a Mac renders the scene and skips
# the comparison below. That still covers kernel compilation, tessellation,
# LaTeX, the fonts and the encoder -- just not the pixels. To gate pixels on a
# Mac, render with ALGAN_UPDATE_FAST_BASELINE=1 there, look at the result, and
# commit it; the comparison turns itself back on, for machines like that one.
BASELINE_KEY = f"macos_{DEVICE}" if sys.platform == "darwin" else DEVICE
EXPECTED_DIR = HERE / f"expected_outputs_{BASELINE_KEY}"
UPDATE_BASELINE = os.getenv("ALGAN_UPDATE_FAST_BASELINE") == "1"


# The one render in the fast suite, and the most expensive thing in it by a
# wide margin. It stays because nothing else in the loop can see the renderer.
pytestmark = pytest.mark.fast

# Free VRAM is what the render loop sizes its frame windows from, and it is not
# reproducible: it shrinks as the Torch and Taichi allocators warm up, so the
# same scene splits differently depending on what ran before it in the process.
# A different split carries a different set of not-yet-spawned actors into the
# batch and pads the merged arrays to a different width, which reorders them and
# the STBVH; silhouettes have moved by up to 54 channel values between two
# splits of one suite. Pinning the measurement makes the render reproducible.
# The figure replaces the measurement rather than capping it, so it has to be
# affordable on the device; 1.5 GiB leaves a 600 MB arena, which this scene
# renders inside in a single batch.
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024


@pytest.fixture
def render_environment(monkeypatch):
    """Isolate the render from the process and pin its frame-window split."""
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


def _load_scene() -> None:
    """Execute the scene file, which records (but never renders) its animation."""
    module_name = "_algan_fast_scene"
    spec = importlib.util.spec_from_file_location(module_name, SCENE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the fast-suite scene {SCENE_FILE}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # The scene module is re-executed per run; never leave it importable.
        sys.modules.pop(module_name, None)


def test_the_fast_scene_renders_and_matches_its_baseline(
    render_environment, assert_video_matches_baseline
):
    output_path = OUTPUT_DIR / "fast.mp4"
    output_path.unlink(missing_ok=True)

    with Scene() as scene:
        _load_scene()
        result = scene.save_video(
            output_path,
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )

    assert result.rendered
    assert result.output_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    expected_path = EXPECTED_DIR / output_path.name
    if UPDATE_BASELINE:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, expected_path)
        pytest.skip(f"re-baselined {output_path.name}")
    if not EXPECTED_DIR.exists():
        pytest.skip(f"no {BASELINE_KEY} fast-suite baseline is available")

    assert expected_path.exists(), (
        "Missing the fast-suite baseline. Re-run with "
        "ALGAN_UPDATE_FAST_BASELINE=1 after reviewing the render."
    )
    assert_video_matches_baseline(
        output_path,
        expected_path,
        HERE / "output_errors" / output_path.name,
        fallback_fps=PREVIEW.frames_per_second,
    )
