"""Shared fixtures and options for the Algan test suites.

Four things live here because they are cross-cutting:

* the ``fast`` marker and ``--fast``, which together define the fast suite:
  the tests marked ``fast`` and nothing else. See ``tests/README.md``;
* a wall-clock report against the fast suite's budget, so the suite cannot
  creep past it unnoticed;
* the frame-by-frame video comparison, shared by ``tests/fast/`` and
  ``tests/full_renders/`` so the two cannot drift apart on tolerance;
* per-test isolation of the two pieces of process-global state Algan owns --
  ``SETTINGS`` and the active-Scene stack. Leaking either between tests makes
  failures depend on test order, which is the hardest kind of flake to chase.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from algan import SETTINGS
from algan.scene_manager import SceneManager

# The render scenes name this family explicitly instead of taking Pango's
# default, because ``Text`` resolves ``font=""`` through fontconfig and the
# glyph advances then change with whatever the machine happens to have
# installed. That is not a hypothetical: the CPU and CUDA baselines were
# rendered on different machines and their Text differed by up to 230 channel
# values -- structurally, not by a sub-pixel shift -- while the geometry and the
# dvisvgm-backed Tex agreed to a mean of 0.36.
#
# Registering the vendored faces makes text depend on bytes in the repository
# rather than on the host, so a container image that changes its fonts can no
# longer look like a renderer regression. See tests/assets/fonts/LICENSE.txt.
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
TEST_FONT = "Algan Test Sans"


def _register_test_fonts():
    """Make the vendored faces visible to Pango for this process."""
    import manimpango

    for face in sorted(FONT_DIR.glob("*.ttf")):
        if not manimpango.register_font(str(face)):
            raise RuntimeError(f"could not register the vendored font {face}")
    if TEST_FONT not in manimpango.list_fonts():
        raise RuntimeError(
            f"registered {FONT_DIR} but Pango still does not offer {TEST_FONT!r}; "
            "the render scenes would silently fall back to a substitute font"
        )


_register_test_fonts()

# The fast suite is meant to stay inside a development loop. This is reported,
# not enforced: the number moves with machine load and thermal state, and a
# timing-based failure would be a flake. A run that exceeds it means the
# curation has drifted -- take a marker off something, rather than raising the
# number reflexively.
#
# The history is worth knowing. The budget was 120 s, then 150 s, while the fast
# suite was *everything not marked slow* and grew with every test anyone added.
# Curating it by hand (only ``fast``-marked tests run) took it from 910 of the
# 1038 collected tests to 191, of which 190 are behavioural and one is the
# render.
# The render is now most of the cost -- it was measured at ~50 s of the old
# 112-147 s suite on CUDA -- so the budget is set to leave it room to pay a
# kernel compile. Raising it again is a deliberate trade of loop time for
# coverage; the point of the number is that growth has to be a decision.
FAST_SUITE_BUDGET_SECONDS = 75.0

# Small per-pixel drift is expected and tolerated: torch CPU rate-function
# evaluation rounds differently depending on the materialization window, so
# byte-identity across re-windowed state is unattainable.
MAX_CHANNEL_DIFFERENCE = 2


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        dest="fast",
        help=(
            "Run only the fast suite: the tests marked 'fast', which are a "
            "hand-picked set covering the mechanisms every change routes "
            "through. Everything else is deselected. See tests/README.md."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Reduce the run to the ``fast`` marker when ``--fast`` is passed.

    Deselected rather than skipped, deliberately: the fast suite excludes most
    of the suite by design, and hundreds of ``s`` characters would bury the
    handful of skips that actually mean something (a missing baseline, an
    absent optional dependency).
    """
    if not config.getoption("fast"):
        return
    selected, deselected = [], []
    for item in items:
        target = selected if item.get_closest_marker("fast") else deselected
        target.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    config._algan_fast_started = time.perf_counter()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Report the fast suite against its budget, so creep is visible."""
    if not config.getoption("fast"):
        return
    started = getattr(config, "_algan_fast_started", None)
    if started is None:
        return
    elapsed = time.perf_counter() - started
    share = elapsed / FAST_SUITE_BUDGET_SECONDS
    line = (
        f"fast suite: {elapsed:.0f}s of its {FAST_SUITE_BUDGET_SECONDS:.0f}s budget "
        f"({share:.0%})"
    )
    if elapsed > FAST_SUITE_BUDGET_SECONDS:
        terminalreporter.write_line(
            line + " -- over budget; take 'fast' off the newest test in it",
            yellow=True,
        )
    else:
        terminalreporter.write_line(line)


@pytest.fixture(autouse=True)
def _isolate_global_settings():
    """Undo any settings a test wrote, however it exited.

    ``SETTINGS`` sections keep stable identity, so ``restore`` writes values
    back into the same objects the engine already holds references to.
    """
    snapshot = SETTINGS.snapshot()
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)


@pytest.fixture
def fresh_scene():
    """A pristine active-Scene stack, torn down again afterwards.

    Prefer ``with Scene() as scene:`` inside a test when you need a handle on
    the Scene; use this when the code under test resolves the *current* Scene
    itself.
    """
    SceneManager.reset()
    try:
        yield SceneManager.instance().current_scene
    finally:
        SceneManager.reset()


@pytest.fixture
def assert_video_matches_baseline():
    """Compare a rendered video to its baseline frame by frame.

    Shared by both render suites so they cannot drift apart on tolerance. A
    frame that exceeds ``MAX_CHANNEL_DIFFERENCE`` on any channel is written to
    ``diff_path`` as a difference video before the assertion fires.
    """
    import cv2
    import numpy as np

    from algan import get_file_writer

    def compare(actual_path, expected_path, diff_path, fallback_fps=10):
        actual = cv2.VideoCapture(str(actual_path))
        expected = cv2.VideoCapture(str(expected_path))
        expected_fps = expected.get(cv2.CAP_PROP_FPS) or fallback_fps
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
                        writer = get_file_writer(
                            str(diff_path),
                            (width, height),
                            codec="libx264rgb",
                            fps=expected_fps,
                            with_mask=False,
                            ffmpeg_params=["-crf", "0", "-preset", "fast"],
                            audiofile=None,
                            audio_codec=None,
                        )
                    writer.write_frame(difference[..., ::-1])
        finally:
            actual.release()
            expected.release()
            if writer is not None:
                writer.close()

        assert frame_count > 0, f"{actual_path.name} did not contain any frames"
        assert max_difference <= MAX_CHANNEL_DIFFERENCE, (
            f"{actual_path.name} differs from its baseline by up to "
            f"{max_difference} channel values (worst at frame {worst_frame}); "
            f"see {diff_path}"
        )

    return compare
