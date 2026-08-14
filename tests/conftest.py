"""Shared fixtures and options for the Algan test suites.

Four things live here because they are cross-cutting:

* the ``slow`` marker and ``--fast`` / ``--skip-slow``, which together define
  the fast suite: everything *not* marked ``slow``. See ``tests/README.md``;
* a wall-clock report against the fast suite's two-minute budget, so the suite
  cannot creep past it unnoticed;
* the frame-by-frame video comparison, shared by ``tests/fast/`` and
  ``tests/full_renders/`` so the two cannot drift apart on tolerance;
* per-test isolation of the two pieces of process-global state Algan owns --
  ``SETTINGS`` and the active-Scene stack. Leaking either between tests makes
  failures depend on test order, which is the hardest kind of flake to chase.
"""

from __future__ import annotations

import time

import pytest

from algan import SETTINGS
from algan.scene_manager import SceneManager

# The fast suite is meant to stay inside a two-and-a-half-minute development
# loop. This is reported, not enforced: the number moves with machine load and
# thermal state, and a timing-based failure would be a flake. A run that exceeds
# it is a prompt to mark the newest expensive test ``slow``.
#
# Raised from 120 s once the behavioural suite had grown past it (419 -> 466
# unit tests), at which point every run reported itself over budget and the
# warning stopped carrying information. Raise it again only after deciding the
# coverage is worth the loop time -- the point of the number is that growth has
# to be a decision.
FAST_SUITE_BUDGET_SECONDS = 150.0

# Small per-pixel drift is expected and tolerated: torch CPU rate-function
# evaluation rounds differently depending on the materialization window, so
# byte-identity across re-windowed state is unattainable.
MAX_CHANNEL_DIFFERENCE = 2


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        "--skip-slow",
        "--skip_slow",
        action="store_true",
        default=False,
        dest="fast",
        help=(
            "Run only the fast suite: everything not marked 'slow'. Targets "
            "under two minutes; see tests/README.md."
        ),
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("fast"):
        return
    skip = pytest.mark.skip(reason="--fast was passed")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


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
            line + " -- over budget; mark the newest expensive test 'slow'",
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

    return compare
