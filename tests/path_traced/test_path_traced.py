"""Pixel-comparison harness for the path-traced scenes.

Each file in ``scenes/`` authors one small Scene that must render through the
``samples_per_pixel > 1`` wavefront path tracer (the scene file sets
``samples_per_pixel`` itself; the harness asserts the plan actually chose the
path tracer, so a scene that silently fell back to the deterministic route
cannot bake a wrong baseline). This module renders each at a deliberately
tiny resolution and compares every frame against the checked-in baseline in
``expected_outputs_<device>/``.

The path tracer promises convergence, not byte-identical frames. It happens
to be stable enough to pixel-compare today -- the Sobol-Owen sampler is a
pure function of path identity, accumulation is atomic-free while no path
splits, and ``available_memory_override`` pins the batching -- so the same
tolerance as the other render suites applies. Should a future change make it
genuinely stochastic run-to-run, this suite moves to a statistical criterion
rather than acquiring a reproducibility requirement. Like ``tests/full_renders``,
the baselines are per *machine*: this suite gates locally and on whichever
machine rendered its baselines, and skips in CI.

Re-baselining
-------------
A rendering change that legitimately alters output is re-baselined by
rendering with the baselines writable and *looking at the result* before
committing it::

    ALGAN_UPDATE_PATH_TRACED_BASELINES=1 .venv/Scripts/python.exe -m pytest \
        tests/path_traced -q

Render twice and commit the second run. On Windows, render work must run one
process at a time: a killed run orphans child processes that keep the output
mp4s locked.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from algan import PREVIEW, SETTINGS, Scene
from algan.scene_manager import SceneManager

HERE = Path(__file__).resolve().parent
SCENES_DIR = HERE / "scenes"
OUTPUT_DIR = HERE / "algan_outputs"
CACHE_DIR = HERE / "algan_cache"
ERRORS_DIR = HERE / "output_errors"
# Device / platform keying as explained at the top of
# ``tests/fast/test_fast_render.py``.
DEVICE = SETTINGS.computing.render_device.type
BASELINE_KEY = f"macos_{DEVICE}" if sys.platform == "darwin" else DEVICE
# Written by a rebaseline; read from only while the mp4s are committed. The
# split is explained in tests/full_renders/test_full_renders.py and the
# resolution order in tests/baseline_store.py.
LOCAL_EXPECTED_DIR = HERE / f"expected_outputs_{BASELINE_KEY}"
UPDATE_BASELINES = os.getenv("ALGAN_UPDATE_PATH_TRACED_BASELINES") == "1"

if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from baseline_store import resolve_baseline_dir  # noqa: E402, I001

LOG_FILE = HERE / "pytest.log"

# Small on purpose: the path tracer pays per (pixel, sample, frame) and this
# suite exists to catch regressions, not to look good. 128x72 at 5 fps with
# one second of animation keeps a scene around a minute on CPU.
PT_SETTINGS = PREVIEW.set(resolution=(128, 72), frames_per_second=5)


class _TeeStream:
    """A stream wrapper that writes to an underlying stream and registered log files."""

    def __init__(self, original_stream: Any) -> None:
        self._orig = original_stream
        self._log_files: list[Any] = []

    def add_file(self, file_obj: Any) -> None:
        if file_obj not in self._log_files:
            self._log_files.append(file_obj)

    def write(self, s: str) -> int:
        res = self._orig.write(s)
        if self._log_files:
            clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)
            for f in list(self._log_files):
                with contextlib.suppress(Exception):
                    f.write(clean)
                    f.flush()
        return res

    def flush(self) -> None:
        self._orig.flush()
        for f in list(self._log_files):
            with contextlib.suppress(Exception):
                f.flush()

    def isatty(self) -> bool:
        return getattr(self._orig, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._orig.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


def _setup_log_piping(log_path: Path, config: Any = None) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    if not hasattr(sys.stdout, "add_file"):
        sys.stdout = _TeeStream(sys.stdout)
    if not hasattr(sys.stderr, "add_file"):
        sys.stderr = _TeeStream(sys.stderr)

    sys.stdout.add_file(log_file)
    sys.stderr.add_file(log_file)

    if config is not None:
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if tr and hasattr(tr, "_tw"):
            if not hasattr(tr._tw._file, "add_file"):
                tr._tw._file = _TeeStream(tr._tw._file)
            tr._tw._file.add_file(log_file)

    atexit.register(log_file.close)
    return log_file


@pytest.fixture(scope="session", autouse=True)
def _pipe_output_to_log(pytestconfig):
    """Pipe all test output, failed assert messages, and success status to a log file."""
    log_file = _setup_log_piping(LOG_FILE, pytestconfig)
    yield
    with contextlib.suppress(Exception):
        log_file.flush()


# Frames are compared by the ``assert_video_matches_baseline`` fixture in
# ``tests/conftest.py``, shared by every render suite so they cannot drift
# apart on tolerance.

# Pinned for the same reason as the other render suites (see
# tests/full_renders/test_full_renders.py): the frame-window split feeds the
# merge order and, for the path tracer, the tile/wave split -- pinning both is
# what lets a stochastic renderer be pixel-compared at all.
AVAILABLE_MEMORY_OVERRIDE = 1536 * 1024 * 1024

# Baselines are per machine, like the full-render suite's (fp32 through a
# path tracer does not survive a libm change; see the measured spread in
# tests/fast/test_fast_render.py). CI covers the path tracer behaviourally
# through tests/unit_tests/test_path_tracer.py; set ALGAN_RUN_PATH_TRACED=1
# to run this suite anyway on a machine whose baselines these are not.
IN_CI = os.getenv("CI") == "true"
FORCE_PATH_TRACED = os.getenv("ALGAN_RUN_PATH_TRACED") == "1"
SKIP_IN_CI_REASON = (
    "path-traced baselines are machine-specific, like the full-render suite's; "
    "CI covers the path tracer through tests/unit_tests/test_path_tracer.py. "
    "Set ALGAN_RUN_PATH_TRACED=1 to override."
)

SCENE_FILES = sorted(
    path for path in SCENES_DIR.glob("*.py") if not path.name.startswith("_")
)


@pytest.fixture(scope="module", autouse=True)
def _clear_output_errors():
    """Empty ``output_errors/`` once, before the first scene of a run."""
    shutil.rmtree(ERRORS_DIR, ignore_errors=True)


@pytest.fixture
def render_environment(monkeypatch):
    """Isolate one scene render from the process and from its neighbours.

    Scenes write ``SETTINGS.raytracing`` (``samples_per_pixel`` at minimum),
    so the whole settings root is snapshotted and restored.

    ``denoise`` is pinned OFF: these baselines gate the ray-traced output
    itself, not a network's smoothing of it (and CI must never depend on
    the denoiser weights being downloadable).
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
    SETTINGS.raytracing.set(denoise=False)
    SceneManager.reset()
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


def _load_scene(scene_path: Path) -> None:
    """Execute a scene file, which records (but never renders) its animation."""
    module_name = f"_algan_path_traced_{scene_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, scene_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load path-traced scene {scene_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Scene modules are re-executed per run; never leave them importable.
        sys.modules.pop(module_name, None)


def test_there_is_at_least_one_path_traced_scene():
    """A silently empty scene directory would make the whole suite vacuous."""
    assert SCENE_FILES, f"no path-traced scenes found in {SCENES_DIR}"


@pytest.mark.skipif(IN_CI and not FORCE_PATH_TRACED, reason=SKIP_IN_CI_REASON)
@pytest.mark.parametrize("scene_path", SCENE_FILES, ids=lambda path: path.stem)
def test_path_traced_scene(
    scene_path: Path, render_environment, assert_video_matches_baseline
):
    output_path = OUTPUT_DIR / f"{scene_path.stem}.mp4"
    output_path.unlink(missing_ok=True)

    with Scene() as scene:
        _load_scene(scene_path)
        result = scene.save_video(
            output_path,
            video_settings=PT_SETTINGS,
            overwrite=True,
            animate_fade_out=False,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )

    assert result.rendered
    assert result.output_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result.render_plan.backend == "path_tracer", (
        f"{scene_path.name} did not render through the path tracer "
        f"(backend {result.render_plan.backend!r}); its baseline would gate "
        f"the wrong renderer"
    )

    if UPDATE_BASELINES:
        LOCAL_EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, LOCAL_EXPECTED_DIR / output_path.name)
        pytest.skip(f"re-baselined {output_path.name}")

    expected_dir = resolve_baseline_dir("path_traced", BASELINE_KEY, LOCAL_EXPECTED_DIR)
    if expected_dir is None:
        pytest.skip(f"no {BASELINE_KEY} path-traced baselines are available")

    expected_path = expected_dir / output_path.name
    assert expected_path.exists(), (
        f"Missing baseline for {scene_path.name}. Re-run with "
        "ALGAN_UPDATE_PATH_TRACED_BASELINES=1 after reviewing the render."
    )
    assert_video_matches_baseline(
        output_path,
        expected_path,
        ERRORS_DIR / output_path.name,
        fallback_fps=PT_SETTINGS.frames_per_second,
    )
